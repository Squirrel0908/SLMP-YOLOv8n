import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager

# --- 1. 环境配置 ---
font_path = '/hy-tmp/SimHei.ttf'
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.family'] = 'sans-serif'

def plot_final_bubble_chart():
    # --- 2. 数据载入 ---
    models = ['Baseline', 'SLM (MLCA)', 'SLC (CDDA)', 'SLCP (Ours)']
    params = [3.01, 1.91, 2.05, 0.84]      # X轴：参数量
    maps = [0.738, 0.768, 0.771, 0.845]    # Y轴：平均鲁棒性mAP
    fps = [1944, 1348, 1302, 1415]         # 气泡大小：FPS
    
    # 颜色配置
    colors = ['#BDC3C7', '#3498DB', '#F39C12', '#C0392B'] # 灰、蓝、橙、红
    
    fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
    
    # --- 3. 绘制气泡 ---
    # 气泡大小缩放系数，根据视觉效果调整
    size_scale = 2 
    scatter = ax.scatter(params, maps, s=np.array(fps)*size_scale, c=colors, 
                        alpha=0.6, edgecolors='white', linewidth=2, zorder=3)
    
    # --- 4. 标注模型名称 ---
    for i, txt in enumerate(models):
        ax.annotate(txt, (params[i], maps[i]), fontsize=14, fontweight='bold',
                    xytext=(0, 15), textcoords='offset points', ha='center')

    # --- 5. 坐标轴美化 ---
    ax.set_xlabel('参数量 (Parameters / M)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel('平均鲁棒精度 (Mean Robust mAP50)', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_title('各阶段模型综合性能评价 (Params vs. Robustness vs. FPS)', fontsize=22, pad=30, fontweight='bold')
    
    # 限制坐标轴，让 SLCP 处于左上方
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0.70, 0.90)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
    
    # --- 6. 添加气泡大小图例 (FPS 示例) ---
    for size in [1000, 1500, 2000]:
        ax.scatter([], [], c='gray', alpha=0.3, s=size*size_scale,
                  label=f'FPS={size}')
    ax.legend(labelspacing=1.5, title="推理速度 (FPS)", loc='lower right', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.savefig('/hy-tmp/Result/Final_Performance_Bubble.pdf', bbox_inches='tight')
    plt.savefig('/hy-tmp/Result/Final_Performance_Bubble.png', dpi=400, bbox_inches='tight')
    print("✅ 终极性能评估气泡图已生成！")

plot_final_bubble_chart()