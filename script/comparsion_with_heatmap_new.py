import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# ================== 参数设置 ==================
root_dir = r"M:\beifen\Sen2_MTC实验结果\postprocess"   # 总目录
gt_folder = "GT"                                           # GT 文件夹名
save_dir = r"E:\研究生\研究生科研学习\paper4_GRSL\comparison_results"

img_exts = (".png", ".jpg", ".jpeg")
font_size = 24
padding = 10
bg_color = (255, 255, 255)

# ================== 新增：展示控制 ==================
exclude_methods = {"DiffCR", "WaveCloudNet"}   # 这些文件夹不展示（按文件夹名精确匹配）
tmfnet_name = "TMFNet"                         # 需要固定到最右边的文件夹名（按文件夹名精确匹配）

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
    error_norm = np.clip((error / 64) * 255, 0, 255).astype(np.uint8)

    # JET 伪彩色
    heatmap = cv2.applyColorMap(error_norm, cv2.COLORMAP_JET)
    heatmap_rgb = heatmap[:, :, ::-1]  # BGR->RGB
    return Image.fromarray(heatmap_rgb)

# ================== 获取所有算法文件夹 ==================
all_folders = [
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f)) and f != gt_folder
]

# ---------- 新增：剔除不展示的方法 ----------
all_folders = [f for f in all_folders if f not in exclude_methods]

# ---------- 新增：TMFNet 固定到最右边 ----------
# 先排序保证稳定
all_folders = sorted(all_folders)

# 如果存在 TMFNet，把它移到最后
if tmfnet_name in all_folders:
    all_folders = [f for f in all_folders if f != tmfnet_name] + [tmfnet_name]
print("Final display order:", ["GT"] + all_folders)

# ================== 主处理流程 ==================
gt_path = os.path.join(root_dir, gt_folder)

for gt_name in os.listdir(gt_path):
    if not gt_name.lower().endswith(img_exts):
        continue

    # ---------- 提取 region 名 ----------
    base_name = os.path.splitext(gt_name)[0]
    region_name = "_".join(base_name.split("_")[:3])
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
            pred_img = Image.open(os.path.join(folder_path, matched)).convert("RGB")

            # 尺寸统一（防止不同算法输出尺寸不一致）
            if pred_img.size != gt_img.size:
                pred_img = pred_img.resize(gt_img.size, Image.BILINEAR)

            err_img = compute_error_heatmap(pred_img, gt_img)

            images_top.append(pred_img)
            images_bottom.append(err_img)
            titles.append(folder)
        else:
            # （可选）不匹配就跳过：保持你原来的逻辑
            # 如果你希望即使缺图也保留空白列，可以告诉我，我给你加占位图
            pass

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
            canvas.paste(bottom, (x_offset, title_height + img_h + padding))

        x_offset += top.width + padding

    # ---------- 保存 ----------
    save_name = f"{region_name}_comparison.png"
    canvas.save(os.path.join(save_dir, save_name))
    print(f"✔ Saved: {save_name}")
