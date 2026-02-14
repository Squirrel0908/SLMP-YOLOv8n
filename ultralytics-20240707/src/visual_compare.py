import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np

# 1. 路径配置（请核对你的绝对路径）
# weights_mlca = '/hy-tmp/runs/train/slm-yolov8n/weights/best.pt'
# weights_cdda = '/hy-tmp/runs/train/slc-yolov8n/weights/best.pt'
weights_slmp = '/hy-tmp/Result/SLMP-yolov8n.pt'
weights_slcp = '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt'

img_dir = '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_weather/images'
save_dir = '/hy-tmp/Result/diff-SLMP_SLCP/comparison_results_weather'
os.makedirs(save_dir, exist_ok=True)

# 2. 加载模型
model_slmp = YOLO(weights_slmp)
model_slcp = YOLO(weights_slcp)

print("🚀 开始自动筛选差异化样本...")

# 3. 遍历图片
img_list = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

for img_name in img_list:
    img_path = os.path.join(img_dir, img_name)
    
    # 推理
    res_slmp = model_slmp(img_path, conf=0.25, verbose=False)[0]
    res_slcp = model_slcp(img_path, conf=0.25, verbose=False)[0]
    
    # 获取检测框数量
    count_slmp = len(res_slmp.boxes)
    count_slcp = len(res_slcp.boxes)
    
    # 筛选逻辑 A：CDDA 发现了 SLMP 没发现的目标（漏检对比）
    is_better_detection = count_slcp > (count_slmp + 1)
    
    # 筛选逻辑 B：置信度大幅提升（即便都检出了，CDDA更自信）
    conf_boost = 0
    if count_slmp > 0 and count_slcp > 0:
        conf_boost = res_slcp.boxes.conf.max() - res_slmp.boxes.conf.max()
    
    # 如果符合任一“优胜”条件，保存对比图
    if is_better_detection or conf_boost > 0.15:
        # 绘制结果
        plot_slmp = res_slmp.plot()
        plot_slcp = res_slcp.plot()
        
        # 拼接图片 (左 SLMP, 右 SLCP)
        combined = np.hstack((plot_slmp, plot_slcp))
        
        # 添加文字标注
        label_slmp = f"SLMP (Count:{count_slmp})"
        label_slcp = f"SLCP (Count:{count_slcp})"
        cv2.putText(combined, label_slmp, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, label_slcp, (plot_slmp.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存
        save_path = os.path.join(save_dir, f"diff_{img_name}")
        cv2.imwrite(save_path, combined)
        print(f"✅ 发现差异图并保存: {img_name} (SLMP:{count_slmp} vs SLCP:{count_slcp})")

print(f"🎉 筛选完成！请去 {save_dir} 目录查看图片。")