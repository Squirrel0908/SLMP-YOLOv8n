import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# ================= 1. 路径与配置 (严格对齐用户目录) =================
paths = {
    'Input':    '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/input',
    'Baseline': '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/baseline',
    'MLCA':     '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/slm',
    'CDDA':     '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/slc'
}

# 工况文件名 (jpg)
row_files = ['Clean.jpg', 'Blur.jpg', 'Illumination.jpg', 'Occlusion.jpg', 'Weather.jpg']

# 标签文本
row_labels = ['原始场景\n(Clean)', '镜头模糊\n(Blur)', '复杂光照\n(Illumination)', '物体遮挡\n(Occlusion)', '雨雾天气\n(Weather)']
col_labels = ['原始图像\n(Input)', 'Baseline\n(YOLOv8n)', 'SLM\n(MLCA)', 'SLC\n(CDDA)']

font_path = '/hy-tmp/SimHei.ttf'
output_dir = '/hy-tmp/Result/Figures_Ch5'
os.makedirs(output_dir, exist_ok=True)

def add_labels(img_np, text, is_top=False, is_left=False):
    """为单张子图添加中英双语标签"""
    img = Image.fromarray(img_np)
    
    # 定义边距
    top_margin = 100 if is_top else 0
    left_margin = 250 if is_left else 0
    
    # 创建新画布
    new_size = (img.width + left_margin, img.height + top_margin)
    canvas = Image.new('RGB', new_size, (255, 255, 255))
    canvas.paste(img, (left_margin, top_margin))
    
    draw = ImageDraw.Draw(canvas)
    
    # 加载字体
    font_main = ImageFont.truetype(font_path, 45)
    font_sub = ImageFont.truetype(font_path, 35)

    if is_top:
        # 在上方居中绘制模型名
        draw.text(((img.width//2) + left_margin, 50), text, font=font_main, fill=(0, 0, 0), anchor="mm")
    
    if is_left:
        # 在左侧居中绘制工况名
        draw.text((125, top_margin + img.height//2), text, font=font_sub, fill=(0, 0, 0), anchor="mm", align="center")
        
    return np.array(canvas)

# ================= 2. 核心拼接逻辑 =================
print("🚀 开始拼接 5x4 热力图矩阵...")
final_matrix_rows = []

for r_idx, filename in enumerate(row_files):
    row_images = []
    for c_idx, (key, folder) in enumerate(paths.items()):
        img_path = os.path.join(folder, filename)
        
        if not os.path.exists(img_path):
            print(f"⚠️ 缺失文件: {img_path}")
            # 如果缺失则生成空白块
            img_np = np.ones((640, 640, 3), dtype=np.uint8) * 255
        else:
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img_np, (640, 640))

        # 处理标签
        processed_img = add_labels(
            img_np, 
            text=col_labels[c_idx] if r_idx == 0 else row_labels[r_idx],
            is_top=(r_idx == 0),
            is_left=(c_idx == 0)
        )
        row_images.append(processed_img)
    
    # 拼接该行
    final_matrix_rows.append(np.hstack(row_images))

# 垂直拼接所有行
full_image = np.vstack(final_matrix_rows)

# 保存
output_file = os.path.join(output_dir, 'Fig5_Heatmap_Comparison_Matrix.png')
Image.fromarray(full_image).save(output_file, dpi=(300, 300))
print(f"✅ 热力对比图已保存至: {output_file}")