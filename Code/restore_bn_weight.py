import torch
import numpy as np
import os

import sys
import warnings
# ==================== 0. 环境与路径设置 ====================
src_path = '/hy-tmp/ultralytics-20240707/src'
if src_path not in sys.path:
    sys.path.append(src_path)

warnings.filterwarnings('ignore')
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO

# ================= 配置区域 =================
# 存放 epoch20.pt, epoch40.pt ... 的文件夹路径
weights_dir = '/hy-tmp/runs/prune/exp5-0.05/slc-yolov8n-groupsl-0.05-prune/weights' 
val_data = '/hy-tmp/ultralytics-20240707/src/dataset/data.yaml' # 你的数据集配置文件
device = 'cuda:0' # 或者 'cpu'

# 存储提取结果
epochs = []
bn_quantiles = []
maps = []

# 获取所有 checkpoint 并排序
pt_files = sorted([f for f in os.listdir(weights_dir) if f.endswith('.pt') and 'best_sl' in f], 
                  key=lambda x: int(''.join(filter(str.isdigit, x))))

print(f"🚀 开始探测 {len(pt_files)} 个 Checkpoints...")

for pt in pt_files:
    epoch_num = int(''.join(filter(str.isdigit, pt)))
    path = os.path.join(weights_dir, pt)
    
    # --- 1. 提取 BN 1% 分位数 ---
    ckpt = torch.load(path, map_location='cpu')
    model_state = ckpt['model'].state_dict()
    
    bn_weights = []
    for key in model_state:
        # 寻找 BN 层的 weight (gamma)
        if 'bn' in key and '.weight' in key:
            bn_weights.extend(model_state[key].abs().numpy().flatten())
    
    quantile_val = np.percentile(bn_weights, 1) # 计算 1% 分位数
    
    # --- 2. 验证获取 mAP ---
    # 使用 YOLO API 进行快速验证
    model = YOLO(path)
    results = model.val(data=val_data, device=device, verbose=False, plots=False)
    map50_95 = results.results_dict['metrics/mAP50-95(B)'] # 获取图 (b) 所需的指标
    
    epochs.append(epoch_num)
    bn_quantiles.append(quantile_val)
    maps.append(map50_95)
    
    print(f"Epoch {epoch_num}: BN_1%={quantile_val:.6f}, mAP={map50_95:.4f}")

# ================= 保存结果 =================
# 你可以将这个数据保存，然后合并到你之前的绘图代码中
import pandas as pd
df = pd.DataFrame({'epoch': epochs, 'bn_1_quantile': bn_quantiles, 'mAP50_95': maps})
df.to_csv('reg_0.05_recovery_log.csv', index=False)
print("✅ 数据还原完成，已保存至 /hy-tmp/runs/prune/exp5-0.05/reg_0.05_recovery_log.csv")