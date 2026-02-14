import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager

# ================= 1. 环境配置 =================
def init_style():
    font_path = '/hy-tmp/SimHei.ttf'
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14

def plot_final_comparison():
    init_style()
    
    # ================= 2. 核心数据 =================
    models = ['Baseline', 'SLM (MLCA)', 'SLC (CDDA)', 'SLCP (Ours)']
    clean_map = [0.938, 0.924, 0.940, 0.927]
    robust_map = [0.738, 0.768, 0.771, 0.845]
    params = [3.01, 1.91, 2.05, 0.84]
    fps = [1944, 1348, 1302, 1415]
    
    x = np.arange(len(models))
    width = 0.35  # 柱子宽度

    # ================= 3. 绘图逻辑 =================
    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)

    # --- 绘制左轴：柱状图 (精度对比) ---
    rects1 = ax1.bar(x - width/2, clean_map, width, label='原始环境精度 (Clean mAP)', color='#BDC3C7', alpha=0.8)
    rects2 = ax1.bar(x + width/2, robust_map, width, label='平均鲁棒精度 (Robust mAP)', color='#C0392B', alpha=0.9)

    ax1.set_ylabel('检测精度 (mAP50)', fontsize=18, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=16, fontweight='bold')
    ax1.set_ylim(0.6, 1.05) # 聚焦在 0.6 以上

    # --- 绘制右轴：折线图 (参数量对比) ---
    ax2 = ax1.twinx()
    ax2.plot(x, params, color='#2E4053', marker='D', markersize=12, linewidth=3, label='参数量 (Parameters)')
    ax2.set_ylabel('参数量 (Millions)', fontsize=18, fontweight='bold', color='#2E4053')
    ax2.set_ylim(0, 4.5)

    # --- 4. 添加标注 (Highlighting) ---
    # 标注压缩率
    ax2.annotate('压缩 72%', xy=(3, 0.84), xytext=(2.2, 0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=14, fontweight='bold')
    
    # 标注鲁棒性增益
    ax1.annotate('鲁棒性提升 14.5%', xy=(3.15, 0.845), xytext=(2.2, 0.78),
                 arrowprops=dict(facecolor='#C0392B', shrink=0.05, width=1),
                 fontsize=14, fontweight='bold', color='#C0392B')

    # 在柱子上方标注 FPS
    for i, f in enumerate(fps):
        ax1.text(x[i], 0.62, f'FPS: {f}', ha='center', va='bottom', fontsize=12, 
                 fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # --- 5. 美化 ---
    plt.title('各阶段改进模型性能与效率综合对比分析', fontsize=24, pad=30, fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1),
               ncol=3, fontsize=14, frameon=True, shadow=True)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('/hy-tmp/Comprehensive_Performance.pdf', bbox_inches='tight')
    plt.savefig('/hy-tmp/Comprehensive_Performance.png', dpi=400, bbox_inches='tight')
    print("✅ 综合性能对比图已生成！")

plot_final_comparison()