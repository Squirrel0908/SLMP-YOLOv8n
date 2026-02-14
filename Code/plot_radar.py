# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import font_manager
# import os

# # ================= 1. 全局风格配置 =================
# def setup_pro_style():
#     font_path = '/hy-tmp/SimHei.ttf'
#     if not os.path.exists(font_path):
#         os.system(f"wget https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf -O {font_path}")
#     font_manager.fontManager.addfont(font_path)
#     plt.rcParams['font.sans-serif'] = ['SimHei']
#     plt.rcParams['axes.unicode_minus'] = False 
#     plt.rcParams['font.family'] = 'sans-serif'
#     # 设置全局英文字体
#     plt.rcParams['font.serif'] = ['Times New Roman']

# def plot_beautified_radar():
#     setup_pro_style()
    
#     # ================= 2. 数据准备 =================
#     labels = ['原始场景\n(Clean)', '运动模糊\n(Blur)', '复杂光照\n(Illumination)', 
#               '物体遮挡\n(Occlusion)', '恶劣天气\n(Weather)']
    
#     # 填入你最新的实验数据
#     models_data = {
#         'YOLOv8n (Baseline)': [0.938, 0.607, 0.888, 0.701, 0.758],
#         'SLM-YOLOv8n (MLCA)': [0.924, 0.656, 0.878, 0.735, 0.805],
#         'SLMP-YOLOv8n (小论文版)': [0.924, 0.644, 0.892, 0.742, 0.790],
#         'SLC-YOLOv8n (CDDA)': [0.940, 0.651, 0.893, 0.759, 0.795],
#         'SLCP-YOLOv8n (本文改进)': [0.927, 0.734, 0.903, 0.858, 0.891]
#     }
    
#     # 配色与线型 (与之前插图保持一致)
#     colors = ['#BDC3C7', '#3498DB', '#6B8FB4', '#F39C12', '#E67E22']
#     linestyles = ['--', '-.', '-.', '--', '-']
#     markers = ['x', 's', 's', 'o', 'o']

#     # ================= 3. 绘图逻辑 =================
#     num_vars = len(labels)
#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1] # 闭合圆环

#     fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True), dpi=300)
    
#     # 调整起始角度，让“原始场景”处于正上方
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)

#     for i, (name, values) in enumerate(models_data.items()):
#         data = values + values[:1]
        
#         # 针对本文模型（SLCP）进行特殊加粗和填充
#         is_ours = '本文改进' in name
#         lw = 4.5 if is_ours else 2.0
#         alpha = 1.0 if is_ours else 0.6
#         zorder = 10 if is_ours else i
        
#         ax.plot(angles, data, color=colors[i], linewidth=lw, linestyle=linestyles[i],
#                 marker=markers[i], markersize=10, label=name, alpha=alpha, zorder=zorder)
        
#         if is_ours:
#             ax.fill(angles, data, color=colors[i], alpha=0.15, zorder=zorder)

#     # ================= 4. 细节打磨 (硕士论文标准) =================
#     # 设置刻度范围
#     ax.set_ylim(0.55, 1.0)
#     ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
#     ax.set_yticklabels(['0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=12, color='gray')
    
#     # 设置网格线风格
#     ax.grid(True, linestyle='--', alpha=0.5)

#     # 设置外圈标签 (解决重叠问题)
#     ax.set_xticks(angles[:-1])
#     # 通过 ha 和 va 精细控制文字位置
#     ax.set_xticklabels(labels, fontsize=15, fontweight='bold', color='#2C3E50')
    
#     # 调整坐标轴标签与图表的距离
#     ax.tick_params(axis='x', pad=25) 

#     # 标题 (加大加粗)
#     plt.title('不同模型在复杂工况下的鲁棒性能对比', fontsize=22, pad=50, fontweight='bold')

#     # 图例 (美化)
#     plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1), fontsize=13, 
#                frameon=True, shadow=True, borderpad=1)

#     plt.tight_layout()
    
#     # 保存
#     save_path = '/hy-tmp/Result/对比模型雷达图/Robustness_Radar_Pro2.pdf'
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.savefig(save_path.replace('.pdf', '.png'), bbox_inches='tight')
#     print(f"✅ 极致美化版雷达图已保存: {save_path}")

# if __name__ == '__main__':
#     plot_beautified_radar()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

# ================= 配置区域 =================
font_path = '/hy-tmp/SimHei.ttf'
output_dir = '/hy-tmp/Result/对比模型雷达图'
os.makedirs(output_dir, exist_ok=True)

# 字体加载
if os.path.exists(font_path):
    font = FontProperties(fname=font_path, size=18)
    legend_font = FontProperties(fname=font_path, size=14)
    title_font = FontProperties(fname=font_path, size=24)
else:
    font = FontProperties(size=18)
    legend_font = FontProperties(size=14)
    title_font = FontProperties(size=24)

# ================= 真实数据录入 (基于截图) =================
labels = np.array(['原始场景\n(Clean)', '运动模糊\n(Blur)', '复杂光照\n(Illum)', '物体遮挡\n(Occ)', '恶劣天气\n(Weather)'])
num_vars = len(labels)

# 数据顺序：Clean, Blur, Illum, Occ, Weather
data = {
    'Baseline (v8n)': {
        'values': [0.9375, 0.6068, 0.8878, 0.7010, 0.7579],
        'color': '#7f7f7f', 'style': '--', 'width': 1.5, 'marker': 'x'
    },
    'SLM (Ch4)': {
        'values': [0.9241, 0.6561, 0.8782, 0.7350, 0.8047],
        'color': '#1f77b4', 'style': '-.', 'width': 1.5, 'marker': '.'
    },
    'SLMP (Slim Pruned)': { 
        'values': [0.9339, 0.6442, 0.8919, 0.7420, 0.7904], # Blur从65降到64，Weather从80降到79
        'color': '#2ca02c', 'style': ':', 'width': 2.0, 'marker': 'v' # 绿色虚线，代表传统剪枝
    },
    'SLC (CDDA Core)': {
        'values': [0.9401, 0.6512, 0.8925, 0.7590, 0.7948],
        'color': '#ff7f0e', 'style': '--', 'width': 2.0, 'marker': 'o'
    },
    'SLCP (Ours Final)': {
        'values': [0.9267, 0.7344, 0.9032, 0.8584, 0.8908], # 冠军数据
        'color': '#d62728', 'style': '-', 'width': 4.0, 'marker': '*' # 红色粗实线
    }
}

# ================= 绘图逻辑 =================
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1] # 闭合

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 绘制各模型
for name, config in data.items():
    values = config['values']
    values += values[:1]
    
    ax.plot(angles, values, 
            color=config['color'], 
            linestyle=config['style'], 
            linewidth=config['width'],
            marker=config['marker'],
            markersize=8,
            label=name)
    
    # 仅填充最终模型，突出包围感
    if 'Ours' in name:
        ax.fill(angles, values, color=config['color'], alpha=0.1)

# ================= 标签与刻度 =================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontproperties=font)

# 动态调整 Y 轴范围，放大差异
plt.ylim(0.55, 0.95)
plt.yticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"], color="grey", size=12)
ax.grid(True, linestyle='--', alpha=0.7)

# 图例与标题
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop=legend_font)
plt.title("五阶段模型全工况鲁棒性演变 (Test Set)", fontproperties=title_font, y=1.08)

# 保存
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Fig6_Radar_5Models_Final.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Fig6_Radar_5Models_Final.pdf'), bbox_inches='tight')

print("✅ 5模型对比雷达图已生成！")
print("观察重点：请注意红色实线 (SLCP) 在 Blur/Occ/Weather 轴上如何显著超越绿色虚线 (SLMP)。")