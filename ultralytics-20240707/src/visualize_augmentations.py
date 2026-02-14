import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics.data.augment import v8_transforms
from ultralytics.cfg import get_cfg


def load_image(image_path):
    """加载图像并返回图像数据"""
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return im


def apply_transforms(image, transforms):
    """将变换应用于图像"""
    # 准备标签字典
    labels = {
        "img": image,
        "cls": np.array([], dtype=np.int32),  # 空的类别数组
        "instances": None,  # 没有实例
        "resized_shape": image.shape[:2]  # 添加调整后的形状
    }
    
    # 应用变换
    transformed = transforms(labels)
    
    return transformed["img"]


def save_augmented_image(image, output_path):
    """保存增强后的图像"""
    cv2.imwrite(output_path, image)


def main():
    parser = argparse.ArgumentParser(description="可视化YOLOv8数据增强效果")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出增强图像文件夹路径")
    parser.add_argument("--image_size", type=int, default=640, help="图像大小")
    parser.add_argument("--n_augmentations", type=int, default=5, help="每张图像生成的增强样本数")
    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取配置
    cfg = get_cfg()
    cfg.imgsz = args.image_size
    
    # 设置数据增强参数
    cfg.mosaic = 1.0  # Mosaic增强概率
    cfg.copy_paste = 0.0  # CopyPaste增强概率
    cfg.degrees = 10.0  # 旋转角度范围
    cfg.translate = 0.1  # 平移比例范围
    cfg.scale = 0.5  # 缩放比例范围
    cfg.shear = 0.0  # 剪切角度范围
    cfg.perspective = 0.0  # 透视变换概率
    cfg.mixup = 0.0  # MixUp增强概率
    cfg.hsv_h = 0.015  # HSV色调调整范围
    cfg.hsv_s = 0.7  # HSV饱和度调整范围
    cfg.hsv_v = 0.4  # HSV亮度调整范围
    cfg.flipud = 0.0  # 垂直翻转概率
    cfg.fliplr = 0.5  # 水平翻转概率

    # 创建模拟数据集对象
    class MockDataset:
        def __init__(self, image):
            self.data = {}
            self.use_segments = False
            self.use_keypoints = False
            self.use_obb = False
            self.imgsz = args.image_size
            self.rect = False
            self.augment = True
            # 添加必要的属性
            self.buffer = [0] * 10  # 包含10个0索引，指向同一个图像
            self.mosaic = 1.0  # 启用mosaic
            self.copy_paste = 0.0  # 禁用copy_paste
            self.image = image  # 保存当前图像
            self.length = 1  # 数据集大小为1
            
        def __len__(self):
            return self.length
            
        def get_image_and_label(self, index):
            # 返回当前图像和空标签
            return {
                "img": self.image.copy(),
                "cls": np.array([], dtype=np.int32),
                "instances": None
            }
            
        def get_indexes(self):
            # 返回有效的索引列表用于mosaic
            return self.buffer

    # 加载并增强图像
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))

    print(f"找到 {len(image_files)} 张图像")

    for i, image_file in enumerate(image_files):
        print(f"处理图像 {i+1}/{len(image_files)}: {image_file.name}")
        
        # 加载图像
        image = load_image(str(image_file))
        
        # 保存原始图像
        original_output_path = output_dir / f"original_{image_file.stem}.jpg"
        save_augmented_image(image, str(original_output_path))
        
        # 为当前图像创建数据集
        dataset = MockDataset(image)
        
        # 创建变换
        transforms = v8_transforms(dataset, args.image_size, cfg)
        
        # 生成增强图像
        for j in range(args.n_augmentations):
            augmented_image = apply_transforms(image, transforms)
            augmented_output_path = output_dir / f"{image_file.stem}_aug_{j+1}.jpg"
            save_augmented_image(augmented_image, str(augmented_output_path))

    print(f"所有图像已处理完成，增强结果保存在: {output_dir}")


if __name__ == "__main__":
    main()