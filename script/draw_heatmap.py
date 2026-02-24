from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# ===================== 这里改参数即可 =====================
# ROOT_DIR   = r"C:\Users\Zhangyu\Desktop\paper4\ppt画图素材\sen2new"
ROOT_DIR   = r"C:\Users\Zhangyu\Desktop\TMFNet\ppt绘图素材\old_dataset"
GT_NAME    = "GT"                      # 关键：GT 是“文件名”，不是文件夹
IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SAVE_EXT   = ".png"                    # 输出热力图扩展名
SUFFIX     = "_comparision"            # 输出名后缀（按你这个拼写）
DEBUG_PRINT = False
# =========================================================


# ================== 误差热力图函数（按你给的形式） ==================
def compute_error_heatmap(pred_img, gt_img):
    pred = np.asarray(pred_img).astype(np.float32)
    gt   = np.asarray(gt_img).astype(np.float32)

    error = np.mean(np.abs(pred - gt), axis=2)
    error_norm = np.clip((error / 64.0) * 255.0, 0, 255).astype(np.uint8)

    if DEBUG_PRINT:
        print("error_norm max/min:", int(error_norm.max()), int(error_norm.min()))

    heatmap_bgr = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    heatmap_rgb = heatmap_bgr[:, :, ::-1]
    return Image.fromarray(heatmap_rgb)
# ==================================================================


def is_image(p: Path) -> bool:
    return p.is_file() and (p.suffix.lower() in IMG_EXTS)


def open_rgb(p: Path) -> Image.Image:
    return Image.open(p).convert("RGB")


def find_gt_file(folder: Path):
    """在 folder 内查找文件名为 GT_NAME 的图像文件（扩展名不限）"""
    for p in folder.iterdir():
        if is_image(p) and p.stem == GT_NAME:
            return p
    return None


# =============== 新增：生成 GT 对应的纯蓝图 =================
def make_solid_blue_like(gt_img: Image.Image) -> Image.Image:
    """返回与 gt_img 同尺寸的纯蓝色 RGB 图 (0,0,255)"""
    w, h = gt_img.size
    return Image.new("RGB", (w, h), (0, 0, 255))
# =========================================================


def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR not found: {root}")

    save_ext = SAVE_EXT.lower() if SAVE_EXT.startswith(".") else "." + SAVE_EXT.lower()

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    matched, saved = 0, 0
    blue_saved = 0

    for sub in subdirs:
        gt_path = find_gt_file(sub)
        if gt_path is None:
            print(f"[SKIP] No GT file in: {sub}")
            continue

        # 读取 GT
        try:
            gt_img = open_rgb(gt_path)
        except Exception as e:
            print(f"[SKIP] GT open failed: {gt_path} | {e}")
            continue

        # ===== 新增：GT 也输出一张纯蓝图 =====
        blue_img = make_solid_blue_like(gt_img)
        gt_blue_path = gt_path.with_name(gt_path.stem + SUFFIX + save_ext)  # GT_comparision.png
        try:
            # 避免重复覆盖也行：如果你想强制覆盖就删掉 exists 判断
            if not gt_blue_path.exists():
                blue_img.save(gt_blue_path)
                blue_saved += 1
                print(f"[OK] GT -> {gt_blue_path.name} (solid blue)")
        except Exception as e:
            print(f"[FAIL] save blue GT: {gt_blue_path} | {e}")

        # 子文件夹内所有其它图像（排除 GT + 排除已生成的 comparision）
        pred_paths = []
        for p in sub.iterdir():
            if not is_image(p):
                continue
            if p.resolve() == gt_path.resolve():
                continue
            if p.stem.endswith(SUFFIX):   # 避免拿热力图再做热力图
                continue
            pred_paths.append(p)

        for pred_path in pred_paths:
            matched += 1
            try:
                pred_img = open_rgb(pred_path)
            except Exception as e:
                print(f"[SKIP] pred open failed: {pred_path} | {e}")
                continue

            # 尺寸不一致：把 pred resize 到 gt
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, resample=Image.BILINEAR)

            heatmap = compute_error_heatmap(pred_img, gt_img)

            out_path = pred_path.with_name(pred_path.stem + SUFFIX + save_ext)
            try:
                heatmap.save(out_path)
                saved += 1
                print(f"[OK] {pred_path.name} vs {gt_path.name} -> {out_path.name}")
            except Exception as e:
                print(f"[FAIL] save: {out_path} | {e}")

    print(f"\nCompared: {matched}, Saved heatmaps: {saved}, Saved GT blue: {blue_saved}")


if __name__ == "__main__":
    main()
