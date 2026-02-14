import os
import cv2
import numpy as np
import random


# ==================== 独立的增强函数 ====================

def random_horizontal_flip(image):
    """
    水平翻转（总是应用，不再随机）
    """
    return cv2.flip(image, 1)  # 总是执行水平翻转，不再随机


def random_vertical_flip(image):
    """
    随机垂直翻转
    """
    if random.random() > 0.5:
        return cv2.flip(image, 0)
    return image.copy()


def random_translation(image, translate_range=0.3):
    """
    随机平移（明显效果）
    :param translate_range: 平移范围，相对于图像尺寸的比例
    """
    h, w = image.shape[:2]
    # 随机平移量，最大为图像尺寸的30%
    tx = random.uniform(-translate_range * w, translate_range * w)
    ty = random.uniform(-translate_range * h, translate_range * h)
    
    # 创建平移矩阵
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # 应用平移，保持图像尺寸不变，边缘用灰色填充
    return cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))


def random_rotation(image, degrees=45.0):
    """
    随机旋转（更大幅度）
    :param degrees: 旋转角度范围，默认为±45度
    """
    h, w = image.shape[:2]
    angle = random.uniform(-degrees, degrees)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(114, 114, 114))


def random_scaling(image, scale_range=(0.5, 2.0)):
    """
    随机缩放（更明显效果）
    :param scale_range: 缩放范围，默认为0.5到2.0（更大的缩放范围）
    """
    h, w = image.shape[:2]
    scale = random.uniform(scale_range[0], scale_range[1])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - (w / 2)
    M[1, 2] += (new_h / 2) - (h / 2)
    
    return cv2.warpAffine(image, M, (new_w, new_h), borderValue=(114, 114, 114))


def adjust_brightness(image, brightness_range=(-0.7, 0.7)):
    """
    随机调整亮度（更明显效果）
    :param brightness_range: 亮度调整范围，默认为-0.7到+0.7（更大的调整范围）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = random.uniform(brightness_range[0], brightness_range[1])
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + brightness), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, contrast_range=(-0.7, 0.7)):
    """
    随机调整对比度（更明显效果）
    :param contrast_range: 对比度调整范围，默认为-0.7到+0.7（更大的调整范围）
    """
    contrast = random.uniform(contrast_range[0], contrast_range[1])
    return np.clip(image * (1 + contrast), 0, 255).astype(np.uint8)


def adjust_saturation(image, saturation_range=(-0.8, 1.5)):
    """
    随机调整饱和度（更明显效果）
    :param saturation_range: 饱和度调整范围，默认为-0.8到+1.5（更大的调整范围）
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = random.uniform(saturation_range[0], saturation_range[1])
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + saturation), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ==================== 组合增强函数 ====================

def augment_image(image, augmentation_methods=None):
    """
    对单个图像应用指定的增强方法
    :param image: 输入图像 (BGR格式)
    :param augmentation_methods: 要应用的增强方法列表
    :return: 增强后的图像
    """
    if augmentation_methods is None:
        # 默认应用所有增强方法
        augmentation_methods = [
            random_horizontal_flip,
            random_vertical_flip,
            random_rotation,
            random_scaling,
            adjust_brightness,
            adjust_contrast,
            adjust_saturation
        ]
    
    augmented = image.copy()
    for method in augmentation_methods:
        augmented = method(augmented)
    
    return augmented.astype(np.uint8)


# ==================== 处理图像并保存 ====================

def process_images(input_dir="test_images_input", output_dir="test_image_output"):
    """
    处理输入目录中的所有图像，对每个图像应用所有增强方法，并按顺序保存每个增强结果
    :param input_dir: 输入图像目录
    :param output_dir: 输出图像目录
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        return
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入目录中的所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"错误：输入目录 {input_dir} 中没有找到图像文件")
        return
    
    # 定义要应用的增强方法列表
    augmentation_methods = [
        ("original", lambda x: x.copy()),  # 原始图像
        ("horizontal_flip", random_horizontal_flip),
        ("vertical_flip", random_vertical_flip),
        ("rotation", random_rotation),
        ("scaling", random_scaling),
        ("brightness", adjust_brightness),
        ("contrast", adjust_contrast),
        ("saturation", adjust_saturation),
        ("all_augmentations", augment_image)  # 所有增强方法的组合
    ]
    
    # 处理每个图像
    for image_index, image_file in enumerate(image_files):
        input_path = os.path.join(input_dir, image_file)
        file_name, file_ext = os.path.splitext(image_file)
        
        try:
            # 读取图像
            original_image = cv2.imread(input_path)
            if original_image is None:
                print(f"警告：无法读取图像 {image_file}")
                continue
            
            print(f"\n处理图像 {image_file} ({image_index + 1}/{len(image_files)})")
            
            # 对每个增强方法应用并保存
            for method_index, (method_name, method_func) in enumerate(augmentation_methods):
                # 应用增强方法
                augmented_image = method_func(original_image)
                
                # 生成输出文件名：[图片索引]_[增强方法索引]_[增强方法名].[扩展名]
                output_file_name = f"{image_index + 1:03d}_{method_index + 1:02d}_{method_name}{file_ext}"
                output_path = os.path.join(output_dir, output_file_name)
                
                # 保存增强后的图像
                cv2.imwrite(output_path, augmented_image)
                print(f"  已保存：{output_file_name} ({method_name})")
            
        except Exception as e:
            print(f"处理图像 {image_file} 时出错：{str(e)}")


if __name__ == "__main__":
    # 默认从test_images_input读取图像，输出到test_image_output
    process_images()
