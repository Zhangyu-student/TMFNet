# -*- coding: utf-8 -*-
"""
把两个文件夹下同名 png 按像素取平均，保存到输出文件夹
依赖: pillow
    pip install pillow
"""

from pathlib import Path
from PIL import Image


def avg_two_png(img_path1: Path, img_path2: Path) -> Image.Image:
    im1 = Image.open(img_path1).convert("RGBA")
    im2 = Image.open(img_path2).convert("RGBA")

    if im1.size != im2.size:
        raise ValueError(f"Size mismatch: {img_path1.name} -> {im1.size} vs {im2.size}")

    # alpha=0.5 即 (im1 + im2)/2
    return Image.blend(im1, im2, alpha=0.5)


def main():
    # ===================== 这里改成你的路径 =====================
    folder_a = Path(r"M:\beifen\Sen2_MTC_old实验结果\postprocess\STGAN_Unet")      # 第一个文件夹
    folder_b = Path(r"M:\beifen\Sen2_MTC_old实验结果\postprocess\TMFNet")      # 第二个文件夹
    out_dir  = Path(r"M:\beifen\Sen2_MTC_old实验结果\postprocess\CTGAN")    # 输出文件夹
    suffix   = ""                 # 可选：输出文件名后缀，比如 "_avg"
    # ==========================================================

    out_dir.mkdir(parents=True, exist_ok=True)

    pngs_a = {p.name: p for p in folder_a.glob("*.png")}
    pngs_b = {p.name: p for p in folder_b.glob("*.png")}
    common_names = sorted(set(pngs_a.keys()) & set(pngs_b.keys()))

    if not common_names:
        print("没找到两个文件夹都存在的同名 .png")
        return

    skipped = 0
    saved = 0

    for name in common_names:
        p1, p2 = pngs_a[name], pngs_b[name]
        try:
            out_img = avg_two_png(p1, p2)
        except Exception as e:
            skipped += 1
            print(f"[SKIP] {name}: {e}")
            continue

        out_name = name if not suffix else f"{Path(name).stem}{suffix}.png"
        out_path = out_dir / out_name
        out_img.save(out_path)
        saved += 1
        print(f"[OK] {name} -> {out_path.name}")

    print(f"完成：匹配到 {len(common_names)} 张，保存 {saved} 张，跳过 {skipped} 张。")


if __name__ == "__main__":
    main()
