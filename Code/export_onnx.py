import torch
import os

import sys
import warnings
# ==================== 0. 环境与路径设置 ====================
src_path = '/hy-tmp/ultralytics-20240707/src'
if src_path not in sys.path:
    sys.path.append(src_path)

warnings.filterwarnings('ignore')
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO
# ================= 配置区域 =================
# 1. 你的冠军模型路径
model_path = '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt'

# 2. 输出目录
output_dir = '/hy-tmp/Result/Export_Models'
os.makedirs(output_dir, exist_ok=True)

def export_for_lubancat():
    print(f"🚀 开始加载模型: {model_path}")
    
    # 加载模型 (必须使用你修改后的源码环境)
    model = YOLO(model_path)
    
    # 导出配置
    # format='onnx' : 导出格式
    # opset=12      : 瑞芯微 RKNN-Toolkit2 对 opset 12 支持最稳健
    # simplify=True : 必须开启！消除 ONNX 中的冗余算子，否则 RKNN 转换易报错
    # imgsz=[640,640]: 瑞芯微 NPU 通常建议使用固定尺寸输入
    
    print("🛠️ 正在执行 ONNX 导出 (含 Simplify)...")
    
    save_path = model.export(
        format='onnx',
        imgsz=[640, 640],
        opset=12,
        simplify=True
    )
    
    print("-" * 50)
    print(f"✅ 导出成功！")
    print(f"ONNX 文件位置: {save_path}")
    print(f"提示: 请将此文件拷贝到安装有 RKNN-Toolkit2 的 PC 环境进行下一步转换。")
    print("-" * 50)

if __name__ == "__main__":
    export_for_lubancat()