import sys
import warnings
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 0. 环境与路径设置 ====================
src_path = '/hy-tmp/ultralytics-20240707/src'
if src_path not in sys.path:
    sys.path.append(src_path)

warnings.filterwarnings('ignore')
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO

# ================= 1. 环境配置 =================
def setup_style():
    font_path = '/hy-tmp/SimHei.ttf'
    from matplotlib import font_manager
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.family'] = 'sans-serif'

def get_spatial_am(feature_map):
    """计算空间激活图 (Spatial Activation Map)"""
    # feature_map shape: [1, C, H, W]
    # 对通道维度取绝对值求和，得到空间分布
    am = torch.sum(torch.abs(feature_map), dim=1).squeeze(0)
    # 归一化到 0-1
    am = (am - am.min()) / (am.max() - am.min() + 1e-8)
    return am.cpu().detach().numpy()

def calculate_similarity(slc_pt, slcp_pt, img_path, layer_idx=9):
    setup_style()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载两个模型
    model_slc = attempt_load_weights(slc_pt, device=device)
    model_slcp = attempt_load_weights(slcp_pt, device=device)
    
    # 2. 图像预处理
    raw_img = cv2.imread(img_path)
    img = cv2.resize(raw_img, (640, 640))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    # 3. 提取特征图 (Hook机制)
    features = {}
    def get_hook(name):
        def hook(model, input, output):
            features[name] = output
        return hook

    # 为两个模型注册 Hook
    model_slc.model[layer_idx].register_forward_hook(get_hook('slc'))
    model_slcp.model[layer_idx].register_forward_hook(get_hook('slcp'))

    # 前向传播
    with torch.no_grad():
        model_slc(img_tensor)
        model_slcp(img_tensor)

    # 4. 计算空间激活图
    am_slc = get_spatial_am(features['slc'])
    am_slcp = get_spatial_am(features['slcp'])

    # 5. 计算余弦相似度 (将其展平为向量对比)
    vec1 = am_slc.flatten()
    vec2 = am_slcp.flatten()
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # 6. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    
    # (a) 原始图片
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'原始工况图像\n({os.path.basename(img_path)})', fontsize=14)
    axes[0].axis('off')

    # (b) SLC 空间激活
    axes[1].imshow(am_slc, cmap='jet')
    axes[1].set_title(f'SLC (未剪枝) 特征分布\n(Params: 2.05M)', fontsize=14)
    axes[1].axis('off')

    # (c) SLCP 空间激活
    axes[2].imshow(am_slcp, cmap='jet')
    axes[2].set_title(f'SLCP (剪枝后) 特征分布\n(Params: 0.84M)', fontsize=14)
    axes[2].axis('off')

    plt.suptitle(f'剪枝前后特征一致性分析 (Feature Fidelity Analysis)\n余弦相似度 (Cosine Similarity): {similarity:.4f}', 
                 fontsize=18, fontweight='bold', y=1.05)
    
    save_path = f'/hy-tmp/Similarity_Check_{layer_idx}.png'
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ 分析完成！相似度: {similarity:.4f}, 图片保存至: {save_path}")

# ================= 3. 执行 =================
if __name__ == '__main__':
    SLC_PT = '/hy-tmp/runs/train/slc-yolov8n/weights/best.pt'
    SLCP_PT = '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt'
    # TEST_IMG = '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_occlusion/images/557.jpg'
    TEST_IMG = '/hy-tmp/ultralytics-20240707/YS_dataset_test/test/images/39.jpg'
    
    calculate_similarity(SLC_PT, SLCP_PT, TEST_IMG, layer_idx=9)