import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shutil
from matplotlib import font_manager
from math import pi

# =================================================================
# ☢️ 核弹级字体修复方案 (请完整复制本块)
# =================================================================
def nuke_font_cache_and_setup():
    # 1. 暴力清除 Matplotlib 缓存 (解决莫名其妙的失效问题)
    cache_dir = matplotlib.get_cachedir()
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"🧹 已清除字体缓存: {cache_dir}")

    # 2. 下载 SimHei 字体
    font_path = '/hy-tmp/SimHei.ttf'
    if not os.path.exists(font_path):
        print("📥 正在下载 SimHei 字体...")
        os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")
    
    # 3. 【核心步骤】将字体文件直接注册到 Matplotlib 内部管理器
    font_manager.fontManager.addfont(font_path)
    
    # 4. 【核心步骤】设置全局默认字体为 SimHei
    # 注意：这里不再强求 Times New Roman，避免服务器没有该字体导致报错
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题
    
    print(f"✅ 全局字体已强制设置为 SimHei (路径: {font_path})")

# 立即执行配置
nuke_font_cache_and_setup()
# =================================================================

# 颜色盘
COLORS = {
    'baseline': '#34495E', 
    '0.001':    '#5DADE2', 
    '0.005':    '#E67E22', 
    '0.01':     '#C0392B', 
    '0.05':     '#27AE60'
}

# =================================================================
# 2. 图一：稀疏训练演变图
# =================================================================
def plot_training_evolution():
    print("正在绘制图一...")
    
    # 路径配置
    configs = {
        '0.0005': {'csv': '/hy-tmp/runs/prune/exp1-0.0005/slc-yolov8n-groupsl-exp1-prune/results.csv', 'log': '/hy-tmp/runs/prune/exp1-0.0005/slc-yolov8n-groupsl-0.0005-50%.log', 'label': r'$\lambda=0.0005$', 'color': '#AED6F1', 'ls': '-'},
        '0.001':  {'csv': '/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.001-prune/results.csv', 'log': '/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.001-50%.log', 'label': r'$\lambda=0.001$', 'color': COLORS['0.001'], 'ls': '-'},
        '0.005':  {'csv': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-prune/results.csv', 'log': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-50%.log', 'label': r'$\lambda=0.005$ (Ours)', 'color': COLORS['0.005'], 'ls': '-'}, # 加粗
        '0.01':   {'csv': '/hy-tmp/runs/prune/exp4-0.01/slc-yolov8n-groupsl-0.01-prune/results.csv', 'log': '/hy-tmp/runs/prune/exp4-0.01/scl-yolov8n-groupsl-0.01-50%.log', 'label': r'$\lambda=0.01$', 'color': COLORS['0.01'], 'ls': '--'}
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    plt.subplots_adjust(wspace=0.15)

    def get_data(c, l):
        w, e_log, m, e_csv = [], [], [], []
        if l and os.path.exists(l):
            with open(l,'r')as f:
                for line in f:
                    if 'bn_weight_1:' in line:
                        try: w.append(float(line.split('bn_weight_1:')[1].split()[0])); e_log.append(int(line.split('epoch:')[1].split()[0]))
                        except: pass
        if c and os.path.exists(c):
            try: df=pd.read_csv(c); df.columns=[x.strip() for x in df.columns]; m=df['metrics/mAP50(B)'].values; e_csv=range(len(m))
            except: pass
        return e_log, w, e_csv, m

    def smooth(y, box_pts=20):
        box = np.ones(box_pts)/box_pts
        return np.convolve(y, box, mode='same')

    for k, v in configs.items():
        el, w, ec, m = get_data(v['csv'], v['log'])
        lw = 3.5 if k == '0.005' else 1.5
        z = 10 if k == '0.005' else 5
        
        if el: ax1.plot(el, w, color=v['color'], lw=lw, ls=v['ls'], label=v['label'], zorder=z)
        if len(m)>0: 
            m_smooth = smooth(m)
            ax2.plot(ec[10:-10], m_smooth[10:-10], color=v['color'], lw=lw, ls=v['ls'], label=v['label'], zorder=z)

    # 注意：这里不需要再传 fontproperties 了，因为全局已经强制设置了 SimHei
    ax1.set_title("(a) BN层权重稀疏化趋势", fontsize=16)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("BN Weight L1 Norm", fontsize=12)
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.legend()

    ax2.set_title("(b) 训练过程 mAP50 演变", fontsize=16)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("mAP@0.5", fontsize=12)
    ax2.set_ylim(0.85, 1.0)
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend(loc='lower right')

    plt.savefig('/hy-tmp/Fig1_Training_Evolution_Final.pdf', bbox_inches='tight')
    print("✅ 图一已保存")

# =================================================================
# 3. 图二：鲁棒性雷达图
# =================================================================
def plot_radar_chart():
    print("正在绘制图二...")
    labels = ['Original', 'Blur', 'Illumination', 'Occlusion', 'Weather']
    data = {
        'Baseline (YOLOv8n)': [0.8354, 0.3928, 0.7770, 0.4879, 0.6188],
        'SLC-Pruned (λ=0.005)': [0.8361, 0.4672, 0.8033, 0.7003, 0.7816]
    }
    
    N = len(labels)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    
    values_base = data['Baseline (YOLOv8n)'] + data['Baseline (YOLOv8n)'][:1]
    ax.plot(angles, values_base, linewidth=2, linestyle='--', color=COLORS['baseline'], label='Baseline (Original)')
    ax.fill(angles, values_base, color=COLORS['baseline'], alpha=0.1)

    values_ours = data['SLC-Pruned (λ=0.005)'] + data['SLC-Pruned (λ=0.005)'][:1]
    ax.plot(angles, values_ours, linewidth=3, linestyle='-', color=COLORS['0.005'], label='Ours (SLC-Pruned)')
    ax.fill(angles, values_ours, color=COLORS['0.005'], alpha=0.2)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12) # 自动使用全局 SimHei
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8], ["0.4", "0.6", "0.8"], color="grey", size=10)
    plt.ylim(0, 0.9)
    
    plt.title("SLC-YOLOv8n vs Baseline 多场景鲁棒性对比", y=1.08, fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    
    plt.savefig('/hy-tmp/Fig2_Robustness_Radar_Final.pdf', bbox_inches='tight')
    print("✅ 图二已保存")

# =================================================================
# 4. 图三：性能衰减对比
# =================================================================
def setup_bilingual_font():
    font_path = '/hy-tmp/SimHei.ttf'
    if not os.path.exists(font_path):
        os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")
    
    # 注册字体
    font_manager.fontManager.addfont(font_path)
    # 创建字体属性对象
    global zh_font, en_font, zh_font_s
    zh_font = font_manager.FontProperties(fname=font_path, size=14)   # 中文标题级
    zh_font_s = font_manager.FontProperties(fname=font_path, size=11) # 中文标签级
    # 设置全局英文为 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False # 解决负号

# =================================================================
# 2. 绘图主逻辑：双语性能保持率对比图
# =================================================================
def plot_bilingual_robustness_retention():
    setup_bilingual_font()
    print("🚀 正在生成中英双语性能保持率对比图...")

    # 数据准备 (中英双语标签)
    scenarios_zh = ['模糊', '光照', '遮挡', '天气']
    scenarios_en = ['Blur', 'Illumination', 'Occlusion', 'Weather']
    labels = [f"{zh}\n{en}" for zh, en in zip(scenarios_zh, scenarios_en)]
    
    # 精度保持率数据 (Retention Rate %)
    retention_baseline = [64.73, 94.7, 74.77, 80.84] # 原始 YOLOv8n
    retention_proposed = [79.25, 97.46, 92.63, 96.13] # 本文算法 SLC-YOLOv8n

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, retention_baseline, width, label='基准模型 / Baseline (YOLOv8n)', 
                    color='#BDC3C7', edgecolor='white', linewidth=0.5, alpha=0.9)
    rects2 = ax.bar(x + width/2, retention_proposed, width, label='本文模型 / Proposed (SLC-YOLOv8n)', 
                    color='#E67E22', edgecolor='white', linewidth=0.5, alpha=1.0)

    # 绘制理想参考线 (100% 理想状态)
    ax.axhline(y=100, color='#2C3E50', linestyle='--', linewidth=1.2, alpha=0.6, 
               label='理想性能 / Ideal Performance (100%)')

    # --- 细节优化 ---
    # 设置标题 (双语)
    ax.set_title("复杂工况下的模型性能保持能力对比\nComparison of Model Performance Retention Capacity under Complex Conditions", 
                 fontproperties=zh_font, pad=25)
    
    # 设置坐标轴 (双语)
    ax.set_ylabel("精度保持率 / mAP Retention Rate (%)", fontproperties=zh_font_s)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=zh_font_s)
    
    # 设置 Y 轴范围和网格
    ax.set_ylim(50, 110)
    ax.grid(axis='y', linestyle=':', alpha=0.5, zorder=0)

    # 设置图例 (双语)
    legend = ax.legend(prop=font_manager.FontProperties(fname='/hy-tmp/SimHei.ttf', size=10), 
                       loc='lower right', frameon=True, edgecolor='black')

    # --- 自动数值标注与提升标注 (核心优化：避免重叠) ---
    def autolabel_with_boost(rects1, rects2):
        for r1, r2 in zip(rects1, rects2):
            h1 = r1.get_height()
            h2 = r2.get_height()
            
            # 标注本文模型的具体数值 (橙色柱子上方)
            ax.annotate(f'{h2:.1f}%',
                        xy=(r2.get_x() + r2.get_width() / 2, h2),
                        xytext=(0, 6),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#D35400')
            
            # 标注提升幅度 (在两个柱子上方空间展示，避免遮挡柱体)
            boost = h2 - h1
            mid_x = (r1.get_x() + r2.get_x() + r2.get_width()) / 2
            max_h = max(h1, h2)
            
            ax.annotate(f'↑ {boost:.1f}%',
                        xy=(mid_x, max_h),
                        xytext=(0, 20), # 将提升幅度标注移得更高，避免与柱子重叠
                        textcoords="offset points",
                        ha='center', va='bottom', 
                        fontsize=10, fontweight='bold', color='#27AE60',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#27AE60", alpha=0.7, lw=0.5))

    autolabel_with_boost(rects1, rects2)

    # 导出文件
    output_pdf = '/hy-tmp/Comparison_Robustness_Bilingual.pdf'
    output_png = '/hy-tmp/Comparison_Robustness_Bilingual.png'
    plt.tight_layout()
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.savefig(output_png, bbox_inches='tight', dpi=300)
    print(f"✅ 绘图任务完成！\nPDF已保存: {output_pdf}\nPNG已保存: {output_png}")

if __name__ == "__main__":
    plot_training_evolution()
    plot_radar_chart()
    plot_bilingual_robustness_retention()
    print("\n🎉 终于搞定了！所有图表已保存至 /hy-tmp/，请查看带 _Final 后缀的文件。")