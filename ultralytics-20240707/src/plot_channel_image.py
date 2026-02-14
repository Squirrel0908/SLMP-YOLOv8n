import os
import warnings
import argparse
import torch
import numpy as np
import torch.nn as nn  # 假设你的环境
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from ultralytics.nn.tasks import attempt_load_weights
# ================= 1. 字体与风格全局标准配置 =================
def setup_pro_style():
    # 字体文件下载路径
    font_path = '/hy-tmp/SimHei.ttf'
    if not os.path.exists(font_path):
        print("正在下载中文字体...")
        os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")
    
    # 核心步骤：将字体注册到全局管理器
    font_manager.fontManager.addfont(font_path)
    # 创建字体属性对象，用于后续手动注入
    font_prop = font_manager.FontProperties(fname=font_path)
    
    # 设置 Seaborn 风格
    sns.set_style("ticks")
    # 设置全局默认字体（作为兜底）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    
    return font_prop

# ================= 2. 核心绘图函数 =================
def plot_thesis_channels(names, base_channels, prune_channels, save_name):
    font_prop = setup_pro_style()
    
    ratios = (np.array(prune_channels) / np.array(base_channels)) * 100
    
    # 设置大论文专用宽幅比例 (22:9)
    fig, ax1 = plt.subplots(figsize=(22, 9))
    x = np.arange(len(names))
    
    # --- 绘制柱状图 (左轴) ---
    # 风格配色：Base(柔和橘金), Prune(学术深红)
    ax1.bar(x, base_channels, color='#e0e0e0', alpha=0.6, label='原始模型通道 (Base)', width=0.75)
    ax1.bar(x, prune_channels, color='#d62728', alpha=0.9, label='剪枝后模型通道 (Pruned)', width=0.75)
    
    # 【核心修复】显式传入 fontproperties
    ax1.set_ylabel('通道数量 (Number of Channels)', fontsize=18, fontweight='bold', fontproperties=font_prop)
    ax1.set_xlabel('网络深度 (卷积层索引)', fontsize=18, fontweight='bold', fontproperties=font_prop)
    ax1.set_ylim(0, max(base_channels) * 1.25)
    
    # --- 绘制保留率折线图 (右轴) ---
    ax2 = ax1.twinx()
    ax2.plot(x, ratios, color='#4b4b4b', linewidth=2, linestyle='--', marker='o', 
             markersize=5, alpha=0.5, label='通道保留率 (%)')
    ax2.set_ylabel('通道保留率 (%)', fontsize=18, fontweight='bold', fontproperties=font_prop)
    ax2.set_ylim(0, 110)
    ax2.axhline(y=50, color='gray', linestyle=':', alpha=0.4) 

    # --- 区域划分标注 ---
    total_len = len(names)
    split_backbone = int(total_len * 0.42)
    split_neck = int(total_len * 0.84)
    
    # 阴影与文字
    ax1.axvspan(0, split_backbone, color='gray', alpha=0.04)
    ax1.text(split_backbone/2, ax1.get_ylim()[1]*0.92, '主干网络 (Backbone)', 
             ha='center', fontsize=18, weight='bold', color='#333333', fontproperties=font_prop)
    
    ax1.axvspan(split_backbone, split_neck, color='blue', alpha=0.03)
    ax1.text((split_backbone+split_neck)/2, ax1.get_ylim()[1]*0.92, '特征融合颈部 (Neck+CDDA)', 
             ha='center', fontsize=18, weight='bold', color='#333333', fontproperties=font_prop)
    
    ax1.text((split_neck+total_len)/2, ax1.get_ylim()[1]*0.92, '检测头 (Head)', 
             ha='center', fontsize=18, weight='bold', color='#333333', fontproperties=font_prop)

    # --- X轴标签稀疏化 ---
    clean_labels = []
    for i, name in enumerate(names):
        if 'CDDA' in name.upper() or 'DETECT' in name.upper() or i % 5 == 0:
            clean_labels.append(name)
        else:
            clean_labels.append("")
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, rotation=60, fontsize=11, ha='right')

    # --- 3. 彻底修复：标题、图例 ---
    # 标题字体加大到 30，且显式注入 font_prop
    ax1.set_title('SLC-YOLOv8n 模型结构压缩(≈50%)与通道保留率分布分析', 
                 fontsize=22, pad=20, fontweight='bold')

    # 图例合并并显式注入 font_prop
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
               prop=font_prop, fontsize=15, frameon=True, shadow=True)

    sns.despine(right=False)
    plt.tight_layout()
    
    # 保存
    plt.savefig(f'{save_name}.png', dpi=400, bbox_inches='tight')
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    print(f"✅ 最终版插图已保存为 {save_name}.pdf")

# ================= 3. 命令行调用逻辑 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-weights', type=str, required=True)
    parser.add_argument('--prune-weights', type=str, required=True)
    parser.add_argument('--output', type=str, default='SLC_Channel_Compare_Final')
    opt = parser.parse_args()
    
    print(f'正在解析模型结构...')
    base_model = attempt_load_weights(opt.base_weights, device=torch.device('cpu'))
    prune_model = attempt_load_weights(opt.prune_weights, device=torch.device('cpu'))
    
    names, base_channels, prune_channels = [], [], []
    
    for (bn, bm), (pn, pm) in zip(base_model.named_modules(), prune_model.named_modules()):
        if isinstance(bm, torch.nn.Conv2d):
            clean_name = bn.replace('model.model.', '').replace('model.', '')
            names.append(clean_name)
            base_channels.append(bm.out_channels)
            prune_channels.append(pm.out_channels)
            
    plot_thesis_channels(names, base_channels, prune_channels, opt.output)