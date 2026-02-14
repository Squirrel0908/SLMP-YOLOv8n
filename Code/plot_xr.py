import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# ================= 1. 环境与字体设置 =================
font_path = '/hy-tmp/SimHei.ttf'
prop = FontProperties(fname=font_path, size=14)
title_p = FontProperties(fname=font_path, size=20)
label_p = FontProperties(fname=font_path, size=15)

# ================= 2. 数据准备 (5工况完整版) =================
labels = ['原始场景', '镜头模糊', '复杂光照', '物体遮挡', '雨雾天气']

# 1. 基准 (Baseline)
v8n_raw = [0.9375, 0.6068, 0.8878, 0.7010, 0.7579]

# 2. 对照组：原生架构 + 融合微调 (数据增强红利)
v8n_pruned_fusion = [0.9280, 0.6680, 0.8950, 0.7620, 0.8150]

# 3. 本文模型 (消融组)：CDDA架构 + 原始集微调 (纯架构贡献)
slcp_base_train = [0.9277, 0.6645, 0.8951, 0.7822, 0.8452]

# 4. 本文模型 (最终版)：CDDA架构 + 融合微调 (协同优化)
slcp_final = [0.9267, 0.7344, 0.9032, 0.8584, 0.8908]

x = np.arange(len(labels))
width = 0.2 

# ================= 3. 开始绘图 =================
plt.figure(figsize=(17, 8.5), dpi=100)

# 统一配色与样式
plt.bar(x - 1.5*width, v8n_raw, width, label='1. Baseline (原生 YOLOv8n)', 
        color='#95a5a6', alpha=0.5, edgecolor='black', hatch='//')

plt.bar(x - 0.5*width, v8n_pruned_fusion, width, label='2. YOLOv8n-Pruned (仅融合微调)', 
        color='#3498db', alpha=0.7, edgecolor='black')

plt.bar(x + 0.5*width, slcp_base_train, width, label='3. SLCP (仅架构改进+原始训练)', 
        color='#f39c12', alpha=0.7, edgecolor='black')

plt.bar(x + 1.5*width, slcp_final, width, label='4. SLCP (最终版: 架构改进+融合微调)', 
        color='#d62728', alpha=0.9, edgecolor='black', linewidth=1.5)

# ================= 4. 关键科学标注 (针对 5 工况) =================
for i in range(1, len(labels)):
    # 最终领先优势 (红色版 vs 蓝色版)
    total_gain = (slcp_final[i] - v8n_pruned_fusion[i]) * 100
    
    # 纯架构增益 (橙色版 vs 灰色基准)
    arch_base_gain = (slcp_base_train[i] - v8n_raw[i]) * 100
    
    # 针对光照环境增益较小，微调标注位置
    y_off = 35 if i != 2 else 15 
    
    # 绘制最终领先标注
    plt.annotate(f'最终领先\n+{total_gain:.1f}%', 
                 xy=(i + 1.5*width, slcp_final[i]), 
                 xytext=(15, y_off), textcoords='offset points',
                 ha='center', va='bottom', fontproperties=FontProperties(fname=font_path, size=11, weight='bold'), 
                 color='#a50000', arrowprops=dict(arrowstyle='->', color='#a50000', lw=1.5))

# ================= 5. 美化坐标轴 =================
plt.ylabel('检测精度 mAP@0.5', fontproperties=label_p)
plt.title('SLCP 模型鲁棒性提升的全要素归因解耦分析 (5 种工况)', fontproperties=title_p, pad=25)
plt.xticks(x, labels, fontproperties=prop)
plt.ylim(0.55, 1.08) # 调高上限，给标注留空间
plt.legend(prop=prop, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=True, shadow=True)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
save_path = '/hy-tmp/Result/Figures_Ch6/Fig6_Attribution_Full_5Conditions.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ 完整 5 工况归因解耦图已生成：{save_path}")