import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.patches as patches
import os

# ================= 1. 环境配置 =================
font_path = '/hy-tmp/SimHei.ttf'
title_p = FontProperties(fname=font_path, size=24)
label_p = FontProperties(fname=font_path, size=20)
legend_p = FontProperties(fname=font_path, size=16)
annot_p = FontProperties(fname=font_path, size=15, weight='bold')

# ================= 2. 数据归档 (确保完全对齐) =================
# 统一标记: Baseline(X), SLM(s), SLC(D), SLMP(^), SLCP(o)
# 数据 [GFLOPs, FPS, Mean_Robust_mAP, Params]
specs = {
    'Baseline':       {'d': [8.1, 203.7, 0.7384, 3.01], 'c': '#7f7f7f', 'm': 'X', 's': 350},
    'SLM (Ch4改进)':   {'d': [4.5, 147.3, 0.7685, 1.91], 'c': '#1f77b4', 'm': 's', 's': 250},
    'SLC (本文母体)':   {'d': [4.6, 120.4, 0.7744, 2.05], 'c': '#ff7f0e', 'm': 'D', 's': 350},
    'SLMP (对照剪枝)':  {'d': [2.2, 142.0, 0.7671, 1.10], 'c': '#2ca02c', 'm': '^', 's': 350},
    'SLCP (最终冠军)':  {'d': [2.3, 116.4, 0.8467, 0.85], 'c': '#d62728', 'm': 'o', 's': 550}
}

# 左图专用的演进序列 (SLCP 1.5x-4.0x)
slcp_x = [3.0, 2.3, 1.8, 1.5, 1.1]
slcp_y = [0.810, 0.8467, 0.841, 0.835, 0.803]
prune_labels = ['1.5x', '2.0x', '2.5x', '3.0x', '4.0x']

# ================= 3. 开始绘图 =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 11), dpi=100)

# --- (a) 左图：理论效率帕累托曲线 ---
ax1.set_facecolor('#fdfdfd')
ax1.axhspan(0.80, 0.93, color='#d62728', alpha=0.03)

# 演化路径虚线
ax1.plot([8.1, 4.6, 3.0], [0.7384, 0.7744, 0.810], color='gray', linestyle=':', linewidth=2.5, alpha=0.5)

# 绘制各点并添加黑色描边
for name, conf in specs.items():
    ax1.scatter(conf['d'][0], conf['d'][2], color=conf['c'], marker=conf['m'], 
                s=conf['s'], edgecolors='black', linewidth=1.5, label=name, zorder=5)

# 绘制红线序列 (描边显著)
ax1.plot(slcp_x, slcp_y, color='#d62728', linewidth=7, marker='o', 
         markersize=16, markerfacecolor='white', markeredgewidth=4, markeredgecolor='#d62728', zorder=6)

# 修正：轻量化方向箭头 (由于X轴反转，向右才是轻量化)
ax1.annotate('轻量化改进方向 (理论算力下降 →)', xy=(1.0, 0.725), xytext=(5.0, 0.725),
             fontproperties=annot_p, color='#2ca02c',
             arrowprops=dict(facecolor='#2ca02c', shrink=0.05, width=5, headwidth=15))

# 标注拐点
ax1.annotate('鲁棒性反转点 (2.0x)', xy=(2.3, 0.8467), xytext=(4.5, 0.89),
             fontproperties=annot_p, arrowprops=dict(arrowstyle='->', color='#d62728', lw=3),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#d62728', alpha=0.9))

# 标注序列
offs = [(-35, 10, 'right'), (0, 35, 'center'), (25, 20, 'left'), (35, 0, 'left'), (0, -45, 'center')]
for i, txt in enumerate(prune_labels):
    ax1.annotate(txt, (slcp_x[i], slcp_y[i]), xytext=offs[i][:2], 
                 textcoords='offset points', ha=offs[i][2], fontproperties=annot_p, color='#a50000')

ax1.invert_xaxis()
ax1.set_xlim(9.5, 0.5); ax1.set_ylim(0.72, 0.95)
ax1.set_xlabel('理论计算量 GFLOPs', fontproperties=label_p)
ax1.set_ylabel('多场景平均鲁棒精度 Mean Robust mAP', fontproperties=label_p)
ax1.set_title("(a) 理论效率：帕累托前沿演进", fontproperties=title_p, pad=25)
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper left', prop=legend_p, frameon=True, shadow=True)

# --- (b) 右图：部署效率评价 (象限迁移图) ---
ax2.set_facecolor('#fdfdfd')
# 红色卓越区
ax2.axhspan(0.80, 0.90, xmin=0, xmax=0.45, color='#d62728', alpha=0.08)
# 灰色基准区 (重新找回)
ax2.axhspan(0.72, 0.76, xmin=0.6, xmax=1.0, color='#7f7f7f', alpha=0.05)

# 绘制各点并统一样式与描边
for name, conf in specs.items():
    ax2.scatter(conf['d'][1], conf['d'][2], color=conf['c'], marker=conf['m'], 
                s=conf['s'], edgecolors='black', linewidth=1.5, zorder=10)

# 箭头：迁移路径 (去除了乱七八糟的红线)
style = "Simple, tail_width=0.5, head_width=8, head_length=12"
a = patches.FancyArrowPatch((195, 0.74), (125, 0.838), connectionstyle="arc3,rad=.15", arrowstyle=style, color="gray", alpha=0.25)
ax2.add_patch(a)
ax2.text(155, 0.81, "鲁棒性跨越式提升\n(Robustness Leap)", rotation=-24, fontproperties=legend_p, color='#444444', ha='center')

# 核心标注
ax2.annotate(f"起点: Baseline", xy=(203.7, 0.7384), xytext=(195, 0.785),
             fontproperties=legend_p, arrowprops=dict(arrowstyle='->', color='gray'))
ax2.annotate(f"最终改进: SLCP", xy=(116.4, 0.8467), xytext=(128, 0.805),
             fontproperties=annot_p, color='#d62728', arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.5))

# 实时参考线
ax2.axvline(x=60, color='gray', linestyle='--', alpha=0.4)
ax2.text(63, 0.77, "工业级实时基准 (60 FPS)", rotation=90, fontproperties=legend_p, color='gray')

# 实时性方向箭头
ax2.annotate('实时性改进方向 (→)', xy=(220, 0.725), xytext=(160, 0.725),
             fontproperties=annot_p, color='#1f77b4',
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=4, headwidth=12))

ax2.set_xlim(50, 240); ax2.set_ylim(0.72, 0.90)
ax2.set_xlabel('实测推理速度 FPS (Batch=1)', fontproperties=label_p)
ax2.set_ylabel('多场景平均鲁棒精度 Mean Robust mAP', fontproperties=label_p)
ax2.set_title("(b) 部署效率：推理速度与稳健性评价", fontproperties=title_p, pad=25)
ax2.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig('/hy-tmp/Result/Figures_Ch6/Fig6_Final_Refined.png', dpi=300)
plt.show()