# -*- coding: utf-8 -*-
"""
TemporalMambaFusionNet (TMFNet) + Ablation Switch (w/o PTF-SSM -> Avg Fusion)

- Input/Output interface matches your WaveDH model:
    input: [B, T, C, H, W]
    out:   [B, output_nc, H, W]

Core idea (default):
- Deep feature fusion uses PTF-SSM (Mamba-like selective scan) per-pixel temporal fusion (C1).
Ablation (w/o PTF-SSM):
- Replace PTF-SSM with temporal average fusion (mean) OR confidence-weighted average (CWTF-style).
- All other parts remain the same (encoder / cloud head / skip CWTF fusion / decoder).

Options:
- use_cloud_head: enable CloudQualityHead to predict sup map (cloud prob). If False, sup=0.
- use_ptf_ssm:    if False -> use TemporalAvgFusion (mean or cwtf), keeping everything else.
- avg_mode:       "mean" or "cwtf" when use_ptf_ssm=False.
"""

from __future__ import annotations
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== add these imports near the top ======
try:
    from ptflops import get_model_complexity_info
    _HAS_PTFLOPS = True
except Exception as e:
    _HAS_PTFLOPS = False
    _PTFLOPS_ERR = e


def count_params_m(model: nn.Module) -> float:
    """Trainable params in Millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


class _ProfileWrapper(nn.Module):
    """
    Wrap your model so ptflops can feed a normal image tensor [B,C,H,W].
    Internally we repeat it along time to [B,T,C,H,W] and return only output image.
    """
    def __init__(self, model: nn.Module, T: int = 3):
        super().__init__()
        self.model = model
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W] -> [B,T,C,H,W]
        x5 = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
        out, _, _ = self.model(x5)
        return out


def profile_macs_g(model: nn.Module, C: int = 3, H: int = 128, W: int = 128, T: int = 3) -> float:
    """
    Return MACs in G (10^9) for one forward pass.
    """
    if not _HAS_PTFLOPS:
        raise RuntimeError(f"ptflops not available: {_PTFLOPS_ERR}")

    wrapper = _ProfileWrapper(model, T=T)
    wrapper.eval()

    # ptflops expects input_res=(C,H,W) for [B,C,H,W]
    macs, params = get_model_complexity_info(
        wrapper,
        (C, H, W),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    # macs: number of MACs (Multiply-Accumulate) for one forward
    return macs / 1e9


# =========================================================
# Basic blocks
# =========================================================
class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        groups: int = 1,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)

        # Make sure num_groups divides out_ch
        ng = min(32, max(1, out_ch // 8))
        while out_ch % ng != 0 and ng > 1:
            ng -= 1
        self.gn = nn.GroupNorm(num_groups=ng, num_channels=out_ch)

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.f = nn.Sequential(
            ConvGNAct(ch, ch, 3, 1),
            ConvGNAct(ch, ch, 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


# =========================================================
# Tiny Selective SSM (PTF-SSM / Mamba-like)
# =========================================================
class TinySelectiveSSMBlock(nn.Module):
    def __init__(self, dim: int, expand: float = 2.0):
        super().__init__()
        hidden = int(dim * expand)

        self.in_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.base_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.sup_to_gate = nn.Sequential(
            nn.Linear(1, dim),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_seq: torch.Tensor, sup_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_seq:   [N, T, C]
        sup_seq: [N, T, 1]  (1=clean, 0=cloudy), optional
        return:  [N, T, C]
        """
        assert x_seq.dim() == 3, f"x_seq must be [N,T,C], got {x_seq.shape}"

        N, T, C = x_seq.shape
        x = self.in_proj(x_seq)

        s = torch.zeros((N, C), device=x.device, dtype=x.dtype)
        ys = []

        for t in range(T):
            xt = x[:, t, :]            # [N,C]
            xt = xt + self.ffn(xt)

            g = self.base_gate(xt)     # [N,C]
            if sup_seq is not None:
                g = g * self.sup_to_gate(sup_seq[:, t, :])  # sup/clean -> gate

            s = (1.0 - g) * s + g * xt
            ys.append(self.out_proj(s))

        y_seq = torch.stack(ys, dim=1)  # [N,T,C]
        return y_seq


class TinyMambaEncoder(nn.Module):
    def __init__(self, d_model: int, depth: int = 2, expand: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([TinySelectiveSSMBlock(d_model, expand=expand) for _ in range(depth)])

    def forward(self, x: torch.Tensor, sup_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, sup_seq)
        return x


# =========================================================
# Per-time encoder + cloud/quality estimator
# =========================================================
class PerTimeEncoderLite(nn.Module):
    """
    Encode one frame x: [B, C, H, W]
    Return:
        f1: [B, base,   H/2, W/2]   (skip1)
        f2: [B, base*2, H/4, W/4]   (skip2)
        f3: [B, base*4, H/4, W/4]   (deep feature)
    """
    def __init__(self, in_ch: int, base: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(in_ch, base, 3, 1),
            ResBlock(base),
        )
        self.down1 = nn.Sequential(
            ConvGNAct(base, base, 3, 2),  # H/2
            ResBlock(base),
        )
        self.down2 = nn.Sequential(
            ConvGNAct(base, base * 2, 3, 2),  # H/4
            ResBlock(base * 2),
        )
        self.deep = nn.Sequential(
            ConvGNAct(base * 2, base * 4, 3, 1),
            ResBlock(base * 4),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x0 = self.stem(x)   # [B, base, H, W]
        f1 = self.down1(x0) # [B, base, H/2, W/2]
        f2 = self.down2(f1) # [B, base*2, H/4, W/4]
        f3 = self.deep(f2)  # [B, base*4, H/4, W/4]
        return f1, f2, f3


class CloudQualityHead(nn.Module):
    """
    Produce a per-frame "cloud/quality" suppression probability map at one scale.
    Input:  x   [B, C, H, W]
    Output: sup [B, 1, H/4, W/4]   (interpret as cloud/suppress prob)
    """
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.s = nn.Sequential(
            ConvGNAct(in_ch, base, 3, 1),
            ConvGNAct(base, base, 3, 2),   # H/2
            ConvGNAct(base, base, 3, 2),   # H/4
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.s(x))  # [B,1,H/4,W/4]


# =========================================================
# Deep fusion modules
# =========================================================
class TemporalMambaFusionC1(nn.Module):
    """
    PTF-SSM fusion (Mamba-like selective scan) per-pixel.
    temporal_features: [B, T, C, H, W]
    sup: [B, T, Hs, Ws] (cloud prob; 1=cloud/suppress)
    Return fused: [B, C, H, W]
    """
    def __init__(self, channels: int, depth: int = 2, expand: float = 2.0, use_target_token: bool = False):
        super().__init__()
        self.channels = channels
        self.use_target_token = use_target_token
        self.temporal = TinyMambaEncoder(d_model=channels, depth=depth, expand=expand)

        self.post = nn.Sequential(
            ConvGNAct(channels, channels, 3, 1),
            ResBlock(channels),
        )

    def forward(self, temporal_features: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = temporal_features.shape

        # sup -> conf (1=clean)
        sup_hw = F.interpolate(sup, size=(H, W), mode="bilinear", align_corners=False)  # [B,T,H,W]
        conf_hw = (1.0 - sup_hw).clamp(0.0, 1.0)  # [B,T,H,W]

        # (optional) explicit suppress on features
        temporal_features = temporal_features * conf_hw.unsqueeze(2)  # [B,T,C,H,W]

        # pixel-wise sequence: [BHW,T,C]
        x = temporal_features.permute(0, 3, 4, 1, 2).contiguous().view(B * H * W, T, C)
        conf_seq = conf_hw.permute(0, 2, 3, 1).contiguous().view(B * H * W, T, 1)

        x = self.temporal(x, conf_seq)  # selective scan

        if self.use_target_token:
            y = x[:, T // 2, :]  # [BHW,C]
        else:
            y = x.mean(dim=1)    # [BHW,C]

        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B,C,H,W]
        return self.post(y)


class TemporalAvgFusion(nn.Module):
    """
    Ablation: replace PTF-SSM with average fusion across time.
    mode:
      - "mean": plain mean over time (ignores sup)
      - "cwtf": confidence-weighted temporal fusion using conf=(1-sup)
    """
    def __init__(self, channels: int, mode: str = "cwtf"):
        super().__init__()
        assert mode in ["mean", "cwtf"]
        self.mode = mode
        self.post = nn.Sequential(
            ConvGNAct(channels, channels, 3, 1),
            ResBlock(channels),
        )

    def forward(self, temporal_features: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = temporal_features.shape

        if self.mode == "mean":
            fused = temporal_features.mean(dim=1)  # [B,C,H,W]
            return self.post(fused)

        # mode == "cwtf"
        sup_hw = F.interpolate(sup, size=(H, W), mode="bilinear", align_corners=False)  # [B,T,H,W]
        conf_hw = (1.0 - sup_hw).clamp(0.0, 1.0)  # [B,T,H,W]

        w = conf_hw / (conf_hw.sum(dim=1, keepdim=True) + 1e-6)      # [B,T,H,W]
        fused = (temporal_features * w.unsqueeze(2)).sum(dim=1)       # [B,C,H,W]
        return self.post(fused)


# =========================================================
# Decoder
# =========================================================
class SimpleDecoder(nn.Module):
    """
    Inputs:
        deep:  [B, C4, H/4, W/4]
        skip2: [B, C2, H/4, W/4]
        skip1: [B, C1, H/2, W/2]
    Output:
        out:   [B, out_ch, H, W]
    """
    def __init__(self, c1: int, c2: int, c4: int, out_ch: int):
        super().__init__()
        self.merge4 = nn.Sequential(
            ConvGNAct(c4 + c2, c4, 3, 1),
            ResBlock(c4),
        )
        self.up2 = nn.Sequential(
            ConvGNAct(c4, c2, 3, 1),
            ResBlock(c2),
        )
        self.merge2 = nn.Sequential(
            ConvGNAct(c2 + c1, c2, 3, 1),
            ResBlock(c2),
        )
        self.up1 = nn.Sequential(
            ConvGNAct(c2, c1, 3, 1),
            ResBlock(c1),
        )
        self.out_head = nn.Sequential(
            ConvGNAct(c1, c1, 3, 1),
            nn.Conv2d(c1, out_ch, 3, padding=1),
        )

    def forward(self, deep: torch.Tensor, skip2: torch.Tensor, skip1: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = out_hw

        x = torch.cat([deep, skip2], dim=1)  # H/4
        x = self.merge4(x)

        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)  # H/2
        x = self.up2(x)

        x = torch.cat([x, skip1], dim=1)  # H/2
        x = self.merge2(x)

        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)  # H
        x = self.up1(x)

        out = self.out_head(x)
        return out


# =========================================================
# Full model
# =========================================================
class TemporalMambaFusionNet(nn.Module):
    """
    forward(input):
        input: [B, T, C, H, W]
        return: out [B, output_nc, H, W], sup_list, None

    Switches:
      - use_ptf_ssm=True  -> deep fusion uses TemporalMambaFusionC1 (PTF-SSM)
      - use_ptf_ssm=False -> deep fusion uses TemporalAvgFusion (mean/cwtf)
    """
    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        base: int = 48,
        temporal_depth: int = 2,
        temporal_expand: float = 2.0,
        use_target_token: bool = False,
        use_cloud_head: bool = True,
        use_ptf_ssm: bool = False,
        avg_mode: str = "mean",   # used only when use_ptf_ssm=False: "mean" or "cwtf"
    ):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.base = base
        self.use_cloud_head = use_cloud_head
        self.use_ptf_ssm = use_ptf_ssm

        self.encoder = PerTimeEncoderLite(in_ch=input_nc, base=base)
        self.cloud_head = CloudQualityHead(in_ch=input_nc, base=32) if use_cloud_head else None

        # Deep fusion: PTF-SSM OR Avg
        if use_ptf_ssm:
            self.fuse_deep = TemporalMambaFusionC1(
                channels=base * 4,
                depth=temporal_depth,
                expand=temporal_expand,
                use_target_token=use_target_token,
            )
        else:
            self.fuse_deep = TemporalAvgFusion(channels=base * 4, mode=avg_mode)

        self.decoder = SimpleDecoder(
            c1=base,
            c2=base * 2,
            c4=base * 4,
            out_ch=output_nc,
        )

    @staticmethod
    def _stack_list(xs: List[torch.Tensor]) -> torch.Tensor:
        # list([B,C,H,W]) -> [B,T,C,H,W]
        return torch.stack(xs, dim=1)

    def _weighted_skip_fuse(self, feats_btchw: torch.Tensor, sup: torch.Tensor) -> torch.Tensor:
        """
        feats_btchw: [B,T,C,H,W]
        sup: [B,T,Hs,Ws]  (cloud prob; 1=cloud/suppress)
        Return: [B,C,H,W]
        """
        B, T, C, H, W = feats_btchw.shape

        sup_u = F.interpolate(
            sup.view(B * T, 1, sup.shape[-2], sup.shape[-1]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, 1, H, W)

        conf = (1.0 - sup_u).clamp(0.0, 1.0)           # [B,T,1,H,W]
        w = conf / (conf.sum(dim=1, keepdim=True) + 1e-6)
        fused = (feats_btchw * w).sum(dim=1)           # [B,C,H,W]
        return fused

    def forward(self, input: torch.Tensor):
        """
        input: [B,T,C,H,W]
        return: out, sup_list, None
        """
        assert input.dim() == 5, f"Expect 5D input [B,T,C,H,W], got {input.shape}"
        B, T, C, H, W = input.shape
        assert C == self.input_nc, f"input_nc mismatch: got C={C}, expect {self.input_nc}"

        f1_list, f2_list, f3_list = [], [], []
        sup_list: List[torch.Tensor] = []

        for t in range(T):
            xt = input[:, t, :, :, :]  # [B,C,H,W]
            f1, f2, f3 = self.encoder(xt)

            if self.use_cloud_head:
                sup = self.cloud_head(xt)  # [B,1,H/4,W/4]
            else:
                sup = torch.zeros(
                    (B, 1, f2.shape[-2], f2.shape[-1]),
                    device=xt.device,
                    dtype=xt.dtype,
                )

            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            sup_list.append(sup)

        f1_bt = self._stack_list(f1_list)  # [B,T,base,H/2,W/2]
        f2_bt = self._stack_list(f2_list)  # [B,T,base*2,H/4,W/4]
        f3_bt = self._stack_list(f3_list)  # [B,T,base*4,H/4,W/4]
        sup_bt = torch.stack(sup_list, dim=1).squeeze(2)  # [B,T,H/4,W/4]

        # Deep fusion: PTF-SSM or Avg (same call signature)
        deep_fused = self.fuse_deep(f3_bt, sup_bt)               # [B,base*4,H/4,W/4]

        # Skip fusion: keep CWTF (confidence-weighted sum) as before
        skip2_fused = self._weighted_skip_fuse(f2_bt, sup_bt)    # [B,base*2,H/4,W/4]
        skip1_fused = self._weighted_skip_fuse(f1_bt, sup_bt)    # [B,base,H/2,W/2]

        out = self.decoder(deep_fused, skip2_fused, skip1_fused, out_hw=(H, W))
        return out, sup_list, None


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 3, 256, 256)  # [B,T,C,H,W]

    # Full model (PTF-SSM) + cloud head
    net_full = TemporalMambaFusionNet(
        input_nc=3, output_nc=3,
        use_cloud_head=True,
        use_ptf_ssm=True
    )

    # Ablation: w/o PTF-SSM -> Avg deep fusion
    net_wo_ptf = TemporalMambaFusionNet(
        input_nc=3, output_nc=3,
        use_cloud_head=True,
        use_ptf_ssm=False,
        avg_mode="cwtf"   # or "mean"
    )

    # ---------- forward sanity ----------
    y_full, sup_full, _ = net_full(x)
    print("[FULL] y:", y_full.shape, "sup0:", sup_full[0].shape)

    y_wo, sup_wo, _ = net_wo_ptf(x)
    print("[w/o PTF-SSM] y:", y_wo.shape, "sup0:", sup_wo[0].shape)

    # ---------- Params + MACs ----------
    for name, net in [("FULL", net_full), ("w/o PTF-SSM", net_wo_ptf)]:
        net.eval()
        p_m = count_params_m(net)

        if _HAS_PTFLOPS:
            macs_g = profile_macs_g(net, C=3, H=256, W=256, T=3)
            print(f"[{name}] Params: {p_m:.3f} M | MACs: {macs_g:.3f} G  (T=3, {256}x{256})")
        else:
            print(f"[{name}] Params: {p_m:.3f} M | MACs: N/A (ptflops not installed)")
            print("Install ptflops by: pip install ptflops")
