# 热力图拼接

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap_comparison(img_path, mlca_path, slc_path, slcp_path, save_name):
    # 1. 加载四张图
    imgs = [cv2.imread(p) for p in [img_path, mlca_path, slc_path, slcp_path]]
    # 统一尺寸 (以原图为准)
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]
    
    # 2. 横向拼接
    combined = np.hstack(imgs)
    
    # 3. 使用 Matplotlib 绘图并添加标题
    plt.figure(figsize=(20, 6))
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    
    # 添加子图标注
    titles = ['(a) 原始工况图像', '(b) MLCA 特征热力图', '(c) SLC (CDDA) 热力图', '(d) SLCP (剪枝微调) 热力图']
    for i, title in enumerate(titles):
        plt.text(w * i + w/2, -20, title, ha='center', fontsize=14, fontweight='bold')
        
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{save_name}.pdf', bbox_inches='tight')
    plt.savefig(f'{save_name}.png', dpi=300, bbox_inches='tight')
    print(f"✅ 热力图对比排版已完成: {save_name}.pdf")

# 调用示例
create_heatmap_comparison('/hy-tmp/Result/MLCA_CDDA_heatmap_diff/input/39-base.jpg', 
                          '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/slm/39-base.jpg', 
                          '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/slc/no-prune/39-base.jpg', 
                          '/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output/slc/0.005-50%/39-base.jpg', 
                          'Fig5_X_Heatmap_Compare')