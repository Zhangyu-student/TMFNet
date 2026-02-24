import os
from PIL import Image, ImageDraw, ImageFont

# ================== 参数设置 ==================
root_dir = r"M:\beifen\Sen2_MTC_old实验结果\postprocess"     # 总目录
gt_folder = "GT"             # GT 文件夹名
save_dir = r"M:\beifen\Sen2_MTC_old实验结果\comparison_results"
# ================== 参数设置 ==================
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

    # 提取地区名称
    base_name = os.path.splitext(gt_name)[0]  # 去掉 .jpg / .png
    region_name = "_".join(base_name.split("_")[:2])
    images = []
    titles = []

    # ---------- 先加入 GT ----------
    gt_img_path = os.path.join(gt_path, gt_name)
    gt_img = Image.open(gt_img_path).convert("RGB")
    images.append(gt_img)
    titles.append("GT")

    # ---------- 加入各算法结果 ----------
    for folder in all_folders:
        folder_path = os.path.join(root_dir, folder)

        matched = None
        for fname in os.listdir(folder_path):
            if fname.endswith(img_exts) and fname.startswith(region_name):
                matched = fname
                break

        if matched:
            img = Image.open(os.path.join(folder_path, matched)).convert("RGB")
            images.append(img)
            titles.append(folder)

    # ---------- 拼接图像 ----------
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths) + padding * (len(images) - 1)
    title_height = font_size + 10
    max_height = max(heights) + title_height

    canvas = Image.new("RGB", (total_width, max_height), bg_color)
    draw = ImageDraw.Draw(canvas)

    x_offset = 0
    for img, title in zip(images, titles):
        # 写标题
        text_w, text_h = draw.textsize(title, font=font)
        draw.text(
            (x_offset + (img.width - text_w) // 2, 0),
            title,
            fill=(0, 0, 0),
            font=font
        )

        # 贴图
        canvas.paste(img, (x_offset, title_height))
        x_offset += img.width + padding

    # ---------- 保存 ----------
    save_name = f"{region_name}_comparison.png"
    canvas.save(os.path.join(save_dir, save_name))
    print(f"✔ Saved: {save_name}")
