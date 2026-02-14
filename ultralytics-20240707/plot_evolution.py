import sys
import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

# ================= 1. 路径注入与环境配置 =================
# 注入源码路径，防止 torch.load 找不到模型类
src_dir = '/hy-tmp/ultralytics-20240707/src' 
if src_dir not in sys.path:
    sys.path.append(src_dir)
font_path = '/hy-tmp/SimHei.ttf'
if not os.path.exists(font_path):
    os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")

# 【核心修复】将字体注册到全局，解决 Glyph 警告
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.sans-serif'] = [prop.get_name()] # 设置全局无衬线字体为黑体
plt.rcParams['axes.unicode_minus'] = False 
# 配置数据读取路径 (0.05 实验)
weights_dir = '/hy-tmp/runs/prune/exp5-0.05/slc-yolov8n-groupsl-0.05-prune/weights'

# 输出路径
save_dir = '/hy-tmp/runs/prune/exp5-0.05'
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'sparsity_evolution_heatmap_cn.png')

# ================= 2. 中文字体自动配置 =================
font_path = '/hy-tmp/SimHei.ttf'
# 如果字体不存在，自动下载
if not os.path.exists(font_path):
    print("正在下载中文字体 SimHei.ttf ...")
    os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")

# 加载字体属性
if os.path.exists(font_path):
    font_prop = FontProperties(fname=font_path)
    print("✅ 中文字体加载成功")
else:
    font_prop = None
    print("⚠️ 警告：字体下载失败，中文可能显示为方块")

# 设置通用绘图风格
plt.rcParams['font.family'] = 'sans-serif' # 英文部分回退
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示

# ================= 3. 数据提取逻辑 =================
print(f"正在扫描目录: {weights_dir}")

if not os.path.exists(weights_dir):
    print(f"❌ 错误：文件夹不存在 -> {weights_dir}")
    exit()

# 筛选 epoch 开头的 pt 文件
pt_files = [f for f in os.listdir(weights_dir) if f.startswith('epoch') and f.endswith('.pt')]

if len(pt_files) == 0:
    print("❌ 错误：未找到以 'epoch' 开头的 .pt 文件！请检查 weights_dir 路径是否正确。")
    exit()
fig, ax = plt.subplots(figsize=(20, 10)) # 稍微调大画布
# 按轮次数字排序 (关键步骤)
pt_files.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
print(f"✅ 成功找到 {len(pt_files)} 个检查点，准备分析...")

data_matrix = []
epochs = []

for pt in pt_files:
    epoch_num = int(re.findall(r'\d+', pt)[0])
    full_path = os.path.join(weights_dir, pt)
    
    try:
        ckpt = torch.load(full_path, map_location='cpu')
        model = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        
        layer_means = []
        # 遍历所有层，提取 BN 权重均值
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                w = m.weight.data.abs().numpy()
                layer_means.append(np.mean(w))
        
        if len(layer_means) > 0:
            data_matrix.append(layer_means)
            epochs.append(epoch_num)
            # print(f"✔️ 已处理 Epoch {epoch_num}")
        
    except Exception as e:
        print(f"❌ 读取 {pt} 失败: {e}")

if len(data_matrix) == 0:
    print("❌ 数据为空，无法绘图。")
    exit()

data_matrix = np.array(data_matrix)
print(f"矩阵构建完成: {data_matrix.shape} (Epochs x Layers)")

# ================= 4. 学术级绘图 (Seaborn) =================
print("正在绘制热力图...")
fig, ax = plt.subplots(figsize=(18, 10))

# 绘制热力图
# cmap='RdYlBu_r': 红(高权重)-黄(中)-蓝(低权重/稀疏)
sns.heatmap(data_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, ax=ax,
            cbar_kws={'label': 'BN层权重均值 (Mean Gamma Value)'})

# --- 核心修复：坐标轴设置 ---

# 1. 标题 (中英双语)
# ax.set_title('SLC-YOLOv8n 层级稀疏演化热力图 ($\lambda=0.005$)', fontsize=20, pad=20, fontproperties=font_prop)
ax.set_title('SLC-YOLOv8n 层级稀疏演化热力图 ($\lambda=0.05$)', fontsize=22, pad=20, fontweight='bold')
# 2. X轴标签
ax.set_xlabel('网络深度 / BN层索引 (Network Depth)', fontsize=16, labelpad=10, fontproperties=font_prop)

# 3. Y轴标签
ax.set_ylabel('训练轮次 (Training Epochs)', fontsize=16, labelpad=10, fontproperties=font_prop)

# 4. 修复 Y轴刻度 (之前报错的地方)
# 确保刻度位于每个单元格的中心，并显示对应的 epoch 数字
ax.set_yticks(np.arange(len(epochs)) + 0.5)
ax.set_yticklabels(epochs, rotation=0, fontsize=12)

# 5. 设置 Colorbar 的字体
cbar = ax.collections[0].colorbar
cbar.set_label('BN层权重均值 (Mean Gamma Value)', fontsize=14, fontproperties=font_prop)

# --- 自动保存 ---
plt.tight_layout()
plt.savefig(save_path, dpi=400, bbox_inches='tight')
print(f"🚀 绘图成功！图片已保存至: {save_path}")