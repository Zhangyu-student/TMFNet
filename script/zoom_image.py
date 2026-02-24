# -*- coding: utf-8 -*-
"""
批量处理文件夹中的图像（修复套娃版）：
- 跳过：文件名包含 'attn' 或 'zoom'
- 对其余图像：画框 + 保存画框图、裁剪ROI
- 关键修复：递归遍历时，自动排除输出目录 out_root（否则会套娃）
依赖：pip install pillow
"""

from pathlib import Path
from PIL import Image, ImageDraw

# =========================================================
# 1) 参数区：你只改这里
# =========================================================

# 输入文件夹
INPUT_DIR = r"C:\Users\Zhangyu\Desktop\TMFNet\ppt绘图素材\T41VMJ_R006_18"

# 输出根目录
# 方案A（推荐）：放到输入目录外面，避免任何风险 & 路径更短
OUTPUT_DIR = r"C:\Users\Zhangyu\Desktop\TMFNet\boxed_outputs_T41VMJ_R006_18"
# 方案B：放输入目录内也可以（因为下面已经排除 out_root，不会套娃）
# OUTPUT_DIR = None  # None -> 默认 INPUT_DIR/boxed_outputs

# 是否递归遍历子文件夹
RECURSIVE = True

# 过滤规则：文件名包含这些关键词就跳过（不区分大小写）
SKIP_KEYWORDS = ["attn", "zoom"]

# 画框样式
BOX_COLOR = (255, 0, 0)   # (R,G,B)
BOX_THICKNESS = 4         # 线宽（像素）

# 框位置与大小（二选一）
USE_ABSOLUTE_BOX = True  # True: 用绝对像素框；False: 用相对比例中心框

# A) 绝对像素框（USE_ABSOLUTE_BOX=True 才生效）
ABS_X = 80
ABS_Y = 50
ABS_W = 70
ABS_H = 70

# B) 相对比例中心框（USE_ABSOLUTE_BOX=False 才生效）
REL_CX = 0.5           # 中心 x ∈ [0,1]
REL_CY = 0.5           # 中心 y ∈ [0,1]
REL_SIZE_RATIO = 0.25  # 边长 = min(W,H)*ratio

# 裁剪子区域保存格式（png/jpg/webp/tif...）
CROP_FORMAT = "png"

# 是否覆盖已存在输出
OVERWRITE = True

# =========================================================
# 2) 脚本主体（一般不改）
# =========================================================

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def should_skip(filename: str) -> bool:
    name = filename.lower()
    return any(k.lower() in name for k in SKIP_KEYWORDS)


def clamp_box(x1, y1, x2, y2, w, h):
    """Clamp box to image boundary. x2,y2 are exclusive for crop()."""
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(1, min(int(x2), w))
    y2 = max(1, min(int(y2), h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def compute_box(img_w, img_h):
    """Return (x1, y1, x2, y2) where x2,y2 are exclusive."""
    if USE_ABSOLUTE_BOX:
        x1, y1 = ABS_X, ABS_Y
        x2, y2 = x1 + ABS_W, y1 + ABS_H
        return clamp_box(x1, y1, x2, y2, img_w, img_h)

    side = int(max(2, round(min(img_w, img_h) * float(REL_SIZE_RATIO))))
    cx = int(round(img_w * float(REL_CX)))
    cy = int(round(img_h * float(REL_CY)))

    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side
    return clamp_box(x1, y1, x2, y2, img_w, img_h)


def draw_box_with_thickness(draw: ImageDraw.ImageDraw, x1, y1, x2, y2, color, thickness: int):
    thickness = max(1, int(thickness))
    for t in range(thickness):
        draw.rectangle([x1 - t, y1 - t, (x2 - 1) + t, (y2 - 1) + t], outline=color)


def iter_images(input_dir: Path, out_root: Path):
    """
    关键修复：
    - 递归遍历时，跳过 out_root 目录下的所有文件（marked/crops/等）
    """
    if RECURSIVE:
        for p in input_dir.rglob("*"):
            if not p.is_file():
                continue
            # ✅ 排除输出目录，防止套娃
            if out_root in p.parents:
                continue
            if p.suffix.lower() in IMG_EXTS:
                yield p
    else:
        for p in input_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                yield p


def main():
    input_dir = Path(INPUT_DIR).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"INPUT_DIR 不存在：{input_dir}")

    out_root = Path(OUTPUT_DIR).expanduser().resolve() if OUTPUT_DIR else (input_dir / "boxed_outputs")
    marked_dir = out_root / "marked"
    crops_dir = out_root / "crops"
    marked_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_name = 0
    skipped_exist = 0

    for img_path in iter_images(input_dir, out_root):
        if should_skip(img_path.name):
            skipped_name += 1
            continue

        # 递归时保持相对目录结构
        rel = img_path.relative_to(input_dir) if RECURSIVE else Path(img_path.name)

        out_marked_path = (marked_dir / rel).with_suffix(img_path.suffix)
        out_crop_path = (crops_dir / rel).with_suffix("." + CROP_FORMAT.lstrip("."))

        out_marked_path.parent.mkdir(parents=True, exist_ok=True)
        out_crop_path.parent.mkdir(parents=True, exist_ok=True)

        if not OVERWRITE and (out_marked_path.exists() or out_crop_path.exists()):
            skipped_exist += 1
            continue

        try:
            img = Image.open(img_path)
            img.load()
        except Exception as e:
            print(f"[WARN] 打开失败: {img_path} ({e})")
            continue

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        W, H = img.size
        x1, y1, x2, y2 = compute_box(W, H)

        roi = img.crop((x1, y1, x2, y2))

        marked = img.copy()
        draw = ImageDraw.Draw(marked)
        draw_box_with_thickness(draw, x1, y1, x2, y2, BOX_COLOR, BOX_THICKNESS)

        try:
            marked.save(out_marked_path)
            roi.save(out_crop_path)
            processed += 1
        except Exception as e:
            print(f"[WARN] 保存失败: {img_path} ({e})")

    print("========== 完成 ==========")
    print(f"processed = {processed}")
    print(f"skipped_by_name(attn/zoom) = {skipped_name}")
    print(f"skipped_existing_outputs = {skipped_exist}")
    print(f"marked_dir = {marked_dir}")
    print(f"crops_dir  = {crops_dir}")


if __name__ == "__main__":
    main()
