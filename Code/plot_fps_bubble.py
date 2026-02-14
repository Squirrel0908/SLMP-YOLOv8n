import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# ================= 1. 环境配置 =================
font_path = '/hy-tmp/SimHei.ttf'
# 调大字号，确保出版清晰
title_p = FontProperties(fname=font_path, size=24)
label_p = FontProperties(fname=font_path, size=18)
annot_p = FontProperties(fname=font_path, size=15)

# ================= 2. 真实数据汇总 =================
# FPS (Batch=1), Mean Robust mAP, Params(M)
data = {
    'Baseline':   {'fps': 203.7, 'map': 0.7384, 'p': 3.01, 'c': '#7f7f7f'},
    'SLM (Ch4)':  {'fps': 147.3, 'map': 0.7685, 'p': 1.91, 'c': '#1f77b4'},
    'SLMP':       {'fps': 142.0, 'map': 0.7671, 'p': 1.10, 'c': '#2ca02c'},
    'SLC (Ch5)':  {'fps': 120.4, 'map': 0.7744, 'p': 2.05, 'c': '#ff7f0e'},
    'SLCP (Ours)':{'fps': 116.4, 'map': 0.8467, 'p': 0.85, 'c': '#d62728'}
}

# ================= 3. 绘图开始 =================
plt.figure(figsize=(12, 8.5), dpi=100)
ax = plt.gca()

# A. 象限背景：淡淡的红色表示“高增益区”
plt.axhspan(0.81, 0.87, xmin=0, xmax=0.3, color='#d62728', alpha=0.05)
plt.text(105, 0.855, "最优性能区\n(高鲁棒+轻量化)", fontproperties=annot_p, color='#a50000', fontweight='bold')

# B. 绘制气泡
# 映射函数：让 0.85M 看起来显著大于 3M，使用指数增强对比
def get_size(p): return (4.0 / p)**2.2 * 300

for name, v in data.items():
    # 突出 SLCP 的边缘
    lw = 4 if name == 'SLCP (Ours)' else 1
    ec = '#a50000' if name == 'SLCP (Ours)' else 'black'
    
    plt.scatter(v['fps'], v['map'], s=get_size(v['p']), color=v['c'], 
                alpha=0.7, edgecolors=ec, linewidths=lw, zorder=10)

# C. 手动调整标签（彻底解决重叠的关键！）
# 每个点的标注：(x_offset, y_offset)
offsets = {
    'Baseline':   (5, 0),
    'SLM (Ch4)':  (0, 8),    # 向上偏移
    'SLMP':       (0, -18),  # 向下偏移，避开 SLM
    'SLC (Ch5)':  (-15, -12),# 向左下偏移
    'SLCP (Ours)':(12, 0)    # 向右偏移
}

for name, v in data.items():
    off = offsets[name]
    plt.annotate(f"{name}\n({v['p']}M)", 
                 xy=(v['fps'], v['map']), 
                 xytext=off,
                 textcoords='offset points',
                 fontproperties=annot_p,
                 fontweight='bold' if 'Ours' in name else 'normal',
                 color='#d62728' if 'Ours' in name else 'black',
                 ha='center' if off[0]==0 else ('left' if off[0]>0 else 'right'))

# D. 装饰性演进线
plt.plot([203, 120], [0.74, 0.835], color='gray', linestyle='--', alpha=0.3, zorder=1)

# E. 气泡图例（做在右下角，不挡数据）
plt.scatter([205], [0.85], s=get_size(0.85), color='none', edgecolors='black', linewidths=1)
plt.scatter([205], [0.835], s=get_size(3.0), color='none', edgecolors='black', linewidths=1)
plt.text(210, 0.85, "轻量化模型 (0.85M)", fontproperties=annot_p, va='center')
plt.text(210, 0.835, "原始模型 (3.0M)", fontproperties=annot_p, va='center')

# ================= 4. 坐标轴美化 =================
plt.xlabel('推理速度 FPS (Batch=1)', fontproperties=label_p)
plt.ylabel('平均鲁棒精度 Mean Robust mAP', fontproperties=label_p)
plt.title('模型效率、参数规模与鲁棒性三维评价', fontproperties=title_p, pad=25)

plt.xlim(100, 225) # 聚焦数据区间，拉开点与点的距离
plt.ylim(0.72, 0.88)
plt.grid(True, linestyle=':', alpha=0.5, zorder=0)

plt.tight_layout()
plt.savefig('/hy-tmp/Result/Figures_Ch6/Fig6_Bubble_Clean_Final.png', dpi=300)
plt.show()