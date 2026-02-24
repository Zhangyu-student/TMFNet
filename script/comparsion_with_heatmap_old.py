import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ================== 参数设置 ==================
root_dir = r"M:\beifen\Sen2_MTC_old实验结果\postprocess"   # 总目录
gt_folder = "GT"                                           # GT 文件夹名
save_dir = r"M:\beifen\Sen2_MTC_old实验结果\comparison_results_old"

img_exts = (".png", ".jpg", ".jpeg")
font_size = 24
padding = 10
bg_color = (255, 255, 255)

os.makedirs(save_dir, exist_ok=True)

# ================== 字体设置 ==================
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except:
    font = ImageFont.load_default()

# ================== 误差热力图函数 ==================
def compute_error_heatmap(pred_img, gt_img):
    """
    pred_img, gt_img: PIL.Image (RGB)
    return: PIL.Image (RGB heatmap)
    """
    pred = np.asarray(pred_img).astype(np.float32)
    gt = np.asarray(gt_img).astype(np.float32)

    # 多通道绝对误差取均值
    error = np.mean(np.abs(pred - gt), axis=2)

    # 归一化到 0–255
    # error_max = np.percentile(error, percentile)
    error_norm = np.clip((error / 64) * 255, 0, 255).astype(np.uint8)
    # JET 伪彩色
    print(error_norm.max(), error_norm.min())
    heatmap = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    # 转换为RGB格式
    heatmap_rgb = heatmap[:, :, ::-1]  # 反转通道顺序
    return Image.fromarray(heatmap_rgb)

# ================== 获取所有算法文件夹 ==================
all_folders = [
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f)) and f != gt_folder
]
all_folders = sorted(all_folders)

# ================== 主处理流程 ==================
gt_path = os.path.join(root_dir, gt_folder)

for gt_name in os.listdir(gt_path):
    if not gt_name.lower().endswith(img_exts):
        continue

    # ---------- 提取 region 名 ----------
    base_name = os.path.splitext(gt_name)[0]
    region_name = "_".join(base_name.split("_")[:2])
    print(f"Processing: {region_name}")

    images_top = []       # 上行：重建结果
    images_bottom = []    # 下行：误差热力图
    titles = []

    # ---------- 加入 GT ----------
    gt_img_path = os.path.join(gt_path, gt_name)
    gt_img = Image.open(gt_img_path).convert("RGB")

    images_top.append(gt_img)
    images_bottom.append(None)   # GT 下方留空
    titles.append("GT")

    # ---------- 加入各算法 ----------
    for folder in all_folders:
        folder_path = os.path.join(root_dir, folder)
        matched = None

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(img_exts):
                continue
            name_wo_ext = os.path.splitext(fname)[0]
            if name_wo_ext.startswith(region_name):
                matched = fname
                break

        if matched:
            pred_img = Image.open(
                os.path.join(folder_path, matched)
            ).convert("RGB")

            # 尺寸统一（防止不同算法输出尺寸不一致）
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.BILINEAR)

            err_img = compute_error_heatmap(pred_img, gt_img)

            images_top.append(pred_img)
            images_bottom.append(err_img)
            titles.append(folder)

    # ---------- 画布尺寸 ----------
    widths = [img.size[0] for img in images_top]
    img_h = images_top[0].size[1]

    title_height = font_size + 10
    total_width = sum(widths) + padding * (len(widths) - 1)
    total_height = title_height + img_h * 2 + padding

    canvas = Image.new("RGB", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    # ---------- 绘制 ----------
    x_offset = 0
    for top, bottom, title in zip(images_top, images_bottom, titles):
        # 标题（Pillow 10+ 安全写法）
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]

        draw.text(
            (x_offset + (top.width - text_w) // 2, 0),
            title,
            fill=(0, 0, 0),
            font=font
        )

        # 上行：重建结果
        canvas.paste(top, (x_offset, title_height))

        # 下行：误差热力图
        if bottom is not None:
            canvas.paste(
                bottom,
                (x_offset, title_height + img_h + padding)
            )

        x_offset += top.width + padding

    # ---------- 保存 ----------
    save_name = f"{region_name}_comparison.png"
    canvas.save(os.path.join(save_dir, save_name))
    print(f"✔ Saved: {save_name}")
