# -*- coding: utf-8 -*-
"""
TemporalMambaFusionNet (NEW MODEL)

- Input/Output interface matches your WaveDH model:
    input: [B, T, C, H, W]
    out:   [B, output_nc, H, W]

Core idea: C1 Pixel-wise temporal sequence fusion (Mamba-like / selective SSM style)
- For each pixel (h, w), take sequence [f1, f2, f3] of length T
- Reshape to [B*H*W, T, C] and run a selective state update (TinySelectiveSSMBlock)
- Use middle token (t = T//2) as fused representation (target-time-centric), then decode.

Option:
- use_cloud_head=False: no CloudQualityHead, set suppression map sup = 0 everywhere (no suppression).
"""

from __future__ import annotations
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.gn = nn.GroupNorm(num_groups=min(32, max(1, out_ch // 8)), num_channels=out_ch)
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


class TinySelectiveSSMBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        hidden = int(dim * expand)

        self.in_proj  = nn.Linear(dim, dim, bias=False)
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

    def forward(self, x_seq: torch.Tensor, sup_seq: torch.Tensor = None):
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
            xt = x[:, t, :]                    # [N,C]
            xt = xt + self.ffn(xt)

            g = self.base_gate(xt)             # [N,C]
            if sup_seq is not None:
                g = g * self.sup_to_gate(sup_seq[:, t, :])  # sup 进关键 gate

            s = (1.0 - g) * s + g * xt
            ys.append(self.out_proj(s))        # 每步输出一个状态

        y_seq = torch.stack(ys, dim=1)         # [N,T,C]
        return y_seq




class TinyMambaEncoder(nn.Module):
    def __init__(self, d_model: int, depth: int = 2, expand: float = 2.0):
        super().__init__()
        self.blocks = nn.ModuleList([TinySelectiveSSMBlock(d_model, expand=expand) for _ in range(depth)])

    def forward(self, x: torch.Tensor, sup_seq: torch.Tensor = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, sup_seq)   # 关键：传进去
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
        f3: [B, base*4, H/4, W/4]   (deep feature, same res as f2 for simplicity)
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
    Input:  x [B, C, H, W]
    Output: sup [B, 1, H/4, W/4]
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
# C1: Pixel-wise temporal fusion (Mamba-like)
# =========================================================
class TemporalMambaFusionC1(nn.Module):
    """
    temporal_features: [B, T, C, H, W]
    sup: [B, T, H4, W4] (suppression prob; 1 means suppress; 0 means keep)
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

        # 1) 先把 sup 上采样到特征分辨率：得到标量 sup_hw: [B,T,H,W]
        sup_hw = F.interpolate(sup, size=(H, W), mode="bilinear", align_corners=False)  # [B,T,H,W]

        # 2) 可选：先用 sup 做一次显式抑制
        temporal_features = temporal_features * (1.0 - sup_hw.unsqueeze(2))  # [B,T,C,H,W]

        # 3) 特征 reshape -> 像素序列 [BHW,T,C]
        x = temporal_features.permute(0, 3, 4, 1, 2).contiguous()  # [B,H,W,T,C]
        x = x.view(B * H * W, T, C)                               # [BHW,T,C]

        # 4) sup reshape -> 像素序列 [BHW,T,1]
        sup_seq = sup_hw.permute(0, 2, 3, 1).contiguous().view(B * H * W, T, 1)  # [BHW,T,1]

        # 5) 传给 temporal
        x = self.temporal(x, sup_seq)                              # [BHW,T,C]

        # 6) 取中间 token / mean
        if self.use_target_token:
            y = x[:, T // 2, :]  # [BHW,C]
        else:
            y = x.mean(dim=1)

        y = y.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()    # [B,C,H,W]
        y = self.post(y)
        return y



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

    If use_cloud_head=False:
        sup is forced to zeros everywhere -> no suppression for any region.
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
    ):
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.base = base
        self.use_cloud_head = use_cloud_head

        self.encoder = PerTimeEncoderLite(in_ch=input_nc, base=base)
        self.cloud_head = CloudQualityHead(in_ch=input_nc, base=32) if use_cloud_head else None

        self.fuse_deep = TemporalMambaFusionC1(
            channels=base * 4,
            depth=temporal_depth,
            expand=temporal_expand,
            use_target_token=use_target_token,
        )

        # NOTE: 你原代码里 gate 实际没用上（注释掉了 gate(sup_u)）
        # 这里保留 module 以便你后面继续扩展；当前融合只用 sup 直接做 conf 权重。
        self.skip_gate_2 = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )
        self.skip_gate_1 = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

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
        sup: [B,T,Hs,Ws]  (Hs,Ws usually H/4,W/4)
        Return: [B,C,H,W]
        """
        B, T, C, H, W = feats_btchw.shape

        # upsample sup to [H,W]
        sup_u = F.interpolate(
            sup.view(B * T, 1, sup.shape[-2], sup.shape[-1]),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, 1, H, W)

        conf = (1.0 - sup_u).clamp(0.0, 1.0)  # [B,T,1,H,W]
        w = conf / (conf.sum(dim=1, keepdim=True) + 1e-6)
        fused = (feats_btchw * w).sum(dim=1)  # [B,C,H,W]
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
                # sup 全 0 => 不抑制任何区域
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

        deep_fused = self.fuse_deep(f3_bt, sup_bt)      # [B,base*4,H/4,W/4]
        skip2_fused = self._weighted_skip_fuse(f2_bt, sup_bt)  # [B,base*2,H/4,W/4]
        skip1_fused = self._weighted_skip_fuse(f1_bt, sup_bt)  # [B,base,H/2,W/2]

        out = self.decoder(deep_fused, skip2_fused, skip1_fused, out_hw=(H, W))
        return out, sup_list, None


# =========================================================
# params & MACs
# =========================================================
if __name__ == "__main__":
    import torch
    from ptflops import get_model_complexity_info

    torch.manual_seed(0)

    # --------- settings ----------
    input_nc  = 3
    output_nc = 3
    T = 3
    H = 256
    W = 256
    device = "cpu"  # 如果你想用GPU可改为 "cuda"
    # ----------------------------

    class OutOnlyWrapper(nn.Module):
        """让 forward 只返回 out，便于 ptflops 统计"""
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, input: torch.Tensor):
            out, _, _ = self.net(input)
            return out

    def input_constructor(input_res):
        """
        ptflops 会把 input_res 原样传进来，我们用它来构造 5D 输入:
        input_res = (T, C, H, W)
        """
        t, c, h, w = input_res
        x = torch.randn(1, t, c, h, w)  # batch=1 统计
        return {"input": x}

    def count_params(model: nn.Module, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        return sum(p.numel() for p in model.parameters())

    def report(use_cloud_head: bool):
        net = TemporalMambaFusionNet(
            input_nc=input_nc,
            output_nc=output_nc,
            use_cloud_head=use_cloud_head,
        ).to(device).eval()

        wrapped = OutOnlyWrapper(net).to(device).eval()

        # 参数量（自己算一遍更稳）
        total_params = count_params(net, trainable_only=False)
        train_params = count_params(net, trainable_only=True)

        # ptflops：MACs & Params
        # 注意：这里 input_res 不是传统 (C,H,W)，而是我们自定义为 (T,C,H,W)
        macs, params = get_model_complexity_info(
            wrapped,
            input_res=(T, input_nc, H, W),
            input_constructor=input_constructor,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False,
        )

        print("=" * 60)
        print(f"use_cloud_head = {use_cloud_head}")
        print(f"Total Params (manual):     {total_params:,}")
        print(f"Trainable Params (manual): {train_params:,}")
        print(f"Params (ptflops):          {params}")
        print(f"MACs  (ptflops):           {macs}")
        print("=" * 60)

    # 统计两种设置（带/不带 cloud head）
    report(use_cloud_head=True)
    report(use_cloud_head=False)
