import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
from scipy.interpolate import make_interp_spline

# =================================================================
# 1. 环境配置与中文字体
# =================================================================
def setup_environment():
    font_path = '/hy-tmp/SimHei.ttf'
    if not os.path.exists(font_path):
        os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")
    
    font_manager.fontManager.addfont(font_path)
    global zh_prop, zh_prop_s
    zh_prop = font_manager.FontProperties(fname=font_path, size=18)
    zh_prop_s = font_manager.FontProperties(fname=font_path, size=14)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False 

# =================================================================
# 2. 增强型数据读取：支持混合来源
# =================================================================
def get_mixed_data(k, info):
    # --- 初始化 ---
    e_log, weights = [], []
    e_csv, maps = [], []

    # --- 情况 A: 处理 0.05 (左图用还原CSV，右图用原始结果CSV) ---
    if k == '0.05':
        # 1. 读取左图数据 (还原出来的 CSV)
        if os.path.exists(info['recovery_csv']):
            df_rec = pd.read_csv(info['recovery_csv'])
            e_log = df_rec['epoch'].values
            weights = df_rec['bn_1_quantile'].values
        
        # 2. 读取右图数据 (原始结果 results.csv)
        if os.path.exists(info['csv']):
            df_res = pd.read_csv(info['csv'])
            df_res.columns = [c.strip() for c in df_res.columns]
            col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df_res.columns else 'metrics/mAP50(B)'
            maps = df_res[col].values
            e_csv = list(range(len(maps)))

    # --- 情况 B: 处理其他版本 (左图用 LOG，右图用结果 CSV) ---
    else:
        # 1. 解析 LOG 获取权重
        if os.path.exists(info['log']):
            with open(info['log'], 'r') as f:
                for line in f:
                    if 'bn_weight_1:' in line:
                        try:
                            weights.append(float(line.split('bn_weight_1:')[1].split()[0]))
                            e_log.append(int(line.split('epoch:')[1].split()[0]))
                        except: continue
        
        # 2. 读取结果 CSV 获取 mAP
        if os.path.exists(info['csv']):
            try:
                df = pd.read_csv(info['csv'])
                df.columns = [c.strip() for c in df.columns]
                col = 'metrics/mAP50-95(B)' if 'metrics/mAP50-95(B)' in df.columns else 'metrics/mAP50(B)'
                maps = df[col].values
                e_csv = list(range(len(maps)))
            except: pass

    return (e_log, weights), (e_csv, maps)

def smooth_recovery_data(x, y):
    """专门为点数稀少的还原数据进行插值平滑"""
    if len(y) < 5: return x, y
    x_new = np.linspace(min(x), max(x), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)
    return x_new, y_smooth

# =================================================================
# 3. 绘图执行
# =================================================================
def plot_analysis():
    setup_environment()
    
    configs = {
        '0.0005': {
            'csv': '/hy-tmp/runs/prune/exp1-0.0005/slc-yolov8n-groupsl-exp1-prune/results.csv', 
            'log': '/hy-tmp/runs/prune/exp1-0.0005/slc-yolov8n-groupsl-0.0005-50%.log',
            'color': '#7f7f7f', 'lw': 1.8, 'ls': '-', 'label': r'$\lambda = 0.0005$'
        },
        '0.001': {
            'csv': '/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.01-prune/results.csv', 
            'log': '/hy-tmp/runs/prune/exp2-0.001/slc-yolov8n-groupsl-0.001-50%.log',
            'color': '#1f77b4', 'lw': 1.8, 'ls': '-', 'label': r'$\lambda = 0.001$'
        },
        '0.01': {
            'csv': '/hy-tmp/runs/prune/exp4-0.01/slc-yolov8n-groupsl-0.01-prune/results.csv', 
            'log': '/hy-tmp/runs/prune/exp4-0.01/scl-yolov8n-groupsl-0.01-50%.log',
            'color': '#ff7f0e', 'lw': 1.8, 'ls': '-', 'label': r'$\lambda = 0.01$'
        },
        '0.05': {
            'recovery_csv': '/hy-tmp/runs/prune/exp5-0.05/reg_0.05_recovery_log.csv',
            'csv': '/hy-tmp/runs/prune/exp5-0.05/slc-yolov8n-groupsl-0.05-prune/results.csv',
            'color': '#272727', 'lw': 2.2, 'ls': '--', 'label': r'$\lambda = 0.05$ (稀疏边界)'
        },
        '0.005': {
            'csv': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-prune/results.csv', 
            'log': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-50%.log',
            'color': '#d62728', 'lw': 4.0, 'ls': '-', 'label': r'$\lambda = 0.005$ (本文选择)'
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), dpi=300)
    
    for k, info in configs.items():
        (e_log, weights), (e_csv, maps) = get_mixed_data(k, info)
        
        # --- (a) 左图：权重收敛速率 ---
        if len(weights) > 0:
            w_arr = np.maximum(0, np.asarray(weights)) # 强制转为 numpy 数组并截断负值
            
            if k == '0.05':
                # 针对点数稀疏的 0.05 使用线性插值，防止样条插值产生负数波动
                x_new = np.linspace(min(e_log), max(e_log), 300)
                y_smooth = np.interp(x_new, e_log, w_arr)
                ax1.plot(x_new, y_smooth, color=info['color'], lw=info['lw'], ls=info['ls'], label=info['label'], zorder=5)
            else:
                # 其他模型正常画，zorder 确保 0.005 在最上层
                ax1.plot(e_log, w_arr, color=info['color'], lw=info['lw'], ls=info['ls'], label=info['label'], zorder=10 if k=='0.005' else 2)

        # --- (b) 右图：模型检测精度稳定性 ---
        if len(maps) > 0:
            m_arr = np.asarray(maps)
            # 使用移动平均平滑，使其曲线更学术化
            y_s = pd.Series(m_arr).rolling(window=5, min_periods=1, center=True).mean()
            
            # 画线
            ax2.plot(e_csv, y_s, color=info['color'], lw=info['lw'], ls=info['ls'], label=info['label'], zorder=10 if k=='0.005' else 2)
            
            # 特殊标注 0.05 的崩溃临界点
            if k == '0.05':
                min_idx = np.argmin(y_s[:120]) # 寻找前期的最低点
                ax2.scatter(e_csv[min_idx], y_s[min_idx], color=info['color'], marker='x', s=180, lw=3, zorder=20)
                ax2.annotate('崩溃临界点', xy=(e_csv[min_idx], y_s[min_idx]), 
                             xytext=(e_csv[min_idx]+40, y_s[min_idx]-0.025),
                             fontproperties=zh_prop_s, color=info['color'], weight='bold',
                             arrowprops=dict(arrowstyle="->", color=info['color'], connectionstyle="arc3,rad=.2"))

    # 统一修饰
    ax1.set_title("(a) 权重收敛速率对比", fontproperties=zh_prop, pad=20)
    ax1.set_xlabel("训练轮次 / Training Epochs", fontproperties=zh_prop_s)
    ax1.set_ylabel("BN权重 1% 分位数", fontproperties=zh_prop_s)
    ax1.grid(True, ls='--', alpha=0.3)
    ax1.legend(prop={'size': 12}, loc='upper right', frameon=True)

    ax2.set_title("(b) 模型检测精度稳定性对比", fontproperties=zh_prop, pad=20)
    ax2.set_xlabel("训练轮次 / Training Epochs", fontproperties=zh_prop_s)
    ax2.set_ylabel("检测精度 / mAP@50-95", fontproperties=zh_prop_s)
    ax2.set_ylim(0.74, 0.96) # 固定 Y 轴范围方便对比
    ax2.grid(True, ls='--', alpha=0.3)
    ax2.legend(prop={'size': 12}, loc='lower right', frameon=True)

    plt.tight_layout()
    plt.savefig("/hy-tmp/Result/SLC_Final_Comparison_Fixed.png", bbox_inches='tight')
    plt.show()

# 别忘了在脚本最下面调用它
if __name__ == "__main__":
    plot_analysis()