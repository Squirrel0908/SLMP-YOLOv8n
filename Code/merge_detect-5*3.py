import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

import sys
import warnings
# ==================== 0. 环境与路径设置 ====================
src_path = '/hy-tmp/ultralytics-20240707/src'
if src_path not in sys.path:
    sys.path.append(src_path)

warnings.filterwarnings('ignore')
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO

# ================= 1. 环境与路径配置 =================
# 输入图片路径 (5张 png)
input_dir = '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/input'
# 权重路径 (请填入你实际的权重路径)
weights = {
    'Baseline': '/hy-tmp/runs/train/yolov8n_baseline_best.pt',
    'SLMP (Ch4)': '/hy-tmp/Result/SLMP-yolov8n.pt',  # 第四章剪枝版
    'SLCP (Ours)': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt' # 第六章最终版
}

# 五种工况的文件名列表 (顺序与雷达图一致)
row_names = ['/hy-tmp/ultralytics-20240707/YS_dataset_test/test/images/290.jpg', 
            '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_blur/images/92.jpg', 
            '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_illumination/images/252.jpg', 
            '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_occlusion/images/87.jpg', 
            '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_weather/images/624.jpg']
# 行标签显示文本 (左侧)
row_labels = ['原始场景\n(Clean)', '镜头模糊\n(Blur)', '复杂光照\n(Illum)', '物体遮挡\n(Occ)', '雨雾天气\n(Weather)']
# 列标签显示文本 (顶部)
col_labels = ['Baseline (v8n)', 'SLMP (Ch4-Slim)', 'SLCP (本文改进)']

font_path = '/hy-tmp/SimHei.ttf'
output_path = '/hy-tmp/Result/Figures_Ch6/Fig6_Detection_Matrix_5x3.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ================= 2. 推理逻辑 =================
print("🚀 开始 5x3 矩阵推理...")
# 存储所有结果图
results_matrix = []

for r_idx, img_name in enumerate(row_names):
    row_results = []
    img_path = os.path.join(input_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"⚠️ 找不到图片: {img_path}，跳过该行。")
        continue

    for c_idx, (m_name, m_path) in enumerate(weights.items()):
        # 加载模型并推理
        model = YOLO(m_path)
        # 强制保存带框图，conf=0.25 保证对比公平性
        res = model.predict(img_path, conf=0.25, save=False, verbose=False)[0]
        
        # 将推理结果转化为 numpy 数组 (带检测框)
        img_bgr = res.plot(labels=True, boxes=True, conf=True) 
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = cv2.resize(img_rgb, (640, 640))
        
        # --- 使用 PIL 绘制学术标签 ---
        pil_img = Image.fromarray(img_res)
        draw = ImageDraw.Draw(pil_img)
        font_main = ImageFont.truetype(font_path, 40)
        font_small = ImageFont.truetype(font_path, 30)

        # 1. 顶部绘制模型名 (仅第一行)
        if r_idx == 0:
            top_bar = Image.new('RGB', (640, 80), (255, 255, 255))
            draw_top = ImageDraw.Draw(top_bar)
            draw_top.text((320, 40), col_labels[c_idx], font=font_main, fill=(0, 0, 0), anchor="mm")
            # 拼接顶部
            new_img = Image.new('RGB', (640, 720))
            new_img.paste(top_bar, (0, 0))
            new_img.paste(pil_img, (0, 80))
            pil_img = new_img

        # 2. 左侧绘制工况名 (仅第一列)
        if c_idx == 0:
            width_with_label = pil_img.width + 200
            left_bar_img = Image.new('RGB', (width_with_label, pil_img.height), (245, 245, 245))
            draw_left = ImageDraw.Draw(left_bar_img)
            # 针对不同工况，SLCP 表现好的地方可以加深底色 (可选)
            draw_left.text((100, pil_img.height//2 + (40 if r_idx==0 else 0)), row_labels[r_idx], 
                           font=font_small, fill=(0, 0, 0), anchor="mm", align="center")
            left_bar_img.paste(pil_img, (200, 0))
            pil_img = left_bar_img
        else:
            # 非第一列需要补齐宽度，保证对齐
            # 如果是第一行，高度已经是 720，否则是 640
            new_img = Image.new('RGB', (640, pil_img.height), (255, 255, 255))
            new_img.paste(pil_img, (0, 0))
            pil_img = new_img

        row_results.append(np.array(pil_img))
    
    # 水平拼接这一行的 3 张图
    final_row = np.hstack(row_results)
    results_matrix.append(final_row)

# 垂直拼接 5 行
final_output = np.vstack(results_matrix)
Image.fromarray(final_output).save(output_path)
print(f"✅ 5x3 矩阵已生成并保存至: {output_path}")