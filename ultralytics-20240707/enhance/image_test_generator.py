import os
import cv2
import numpy as np
import random


# ==================== 测试集生成函数 ====================

def apply_motion_blur(image):
    """
    模拟运动模糊
    :param image: 输入图像 (BGR格式)
    :return: 运动模糊处理后的图像
    """
    # 选择核大小k ∈ {7, 15}
    kernel_sizes = [7, 15]
    kernel_size = random.choice(kernel_sizes)
    
    # 随机选择角度
    angle = random.randint(0, 180)
    
    # 创建运动模糊核
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = 1.0
    
    # 旋转核
    rotation_matrix = cv2.getRotationMatrix2D((kernel_size // 2, kernel_size // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
    
    # 归一化
    kernel = kernel / kernel_size
    
    # 应用模糊
    blurred = cv2.filter2D(image, -1, kernel)
    
    return blurred


def apply_occlusion(image):
    """
    模拟随机块遮挡（Cutout）
    :param image: 输入图像 (BGR格式)
    :return: 遮挡处理后的图像
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # 计算目标区域面积（10%-40%的图像面积）
    image_area = h * w
    target_area = random.uniform(0.1, 0.4) * image_area
    
    # 生成1-3个遮挡块
    num_blocks = random.randint(1, 3)
    
    for _ in range(num_blocks):
        # 随机生成块的大小和位置
        block_width = random.randint(20, int(w * 0.5))
        block_height = random.randint(20, int(h * 0.5))
        
        # 确保块不会超出图像边界
        x = random.randint(0, max(0, w - block_width))
        y = random.randint(0, max(0, h - block_height))
        
        # 随机选择遮挡类型：黑色块或噪声块
        if random.random() > 0.5:
            # 黑色块
            result[y:y+block_height, x:x+block_width] = 0
        else:
            # 噪声块
            noise = np.random.randint(0, 256, (block_height, block_width, 3), dtype=np.uint8)
            result[y:y+block_height, x:x+block_width] = noise
    
    return result


def apply_extreme_illumination(image):
    """
    模拟极端光照（过曝或阴影）
    :param image: 输入图像 (BGR格式)
    :return: 光照处理后的图像
    """
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 随机选择光照类型：过曝或阴影
    if random.random() > 0.5:
        # 过曝：增加亮度
        brightness_factor = random.uniform(1.5, 3.0)  # 1.5-3.0倍亮度
    else:
        # 阴影：降低亮度
        brightness_factor = random.uniform(0.1, 0.5)  # 0.1-0.5倍亮度
    
    # 调整V通道（亮度）
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
    
    # 添加随机阴影（可选）
    if random.random() > 0.5:
        h, w = image.shape[:2]
        num_shadows = random.randint(1, 2)
        
        for _ in range(num_shadows):
            # 创建阴影区域
            shadow_height = random.randint(int(h * 0.2), int(h * 0.8))
            shadow_width = random.randint(int(w * 0.2), int(w * 0.8))
            
            # 随机位置
            x = random.randint(0, w - shadow_width)
            y = random.randint(0, h - shadow_height)
            
            # 随机阴影强度
            shadow_intensity = random.uniform(0.3, 0.8)
            
            # 直接在hsv数组上应用阴影，避免创建临时数组导致的问题
            hsv[y:y+shadow_height, x:x+shadow_width, 2] = np.clip(
                hsv[y:y+shadow_height, x:x+shadow_width, 2] * shadow_intensity,
                0, 255
            ).astype(np.uint8)
    
    # 转换回BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return result


# ==================== 主处理函数 ====================

def generate_test_sets(input_dir, output_dir):
    """
    生成三个测试集：遮挡测试集、模糊测试集和光照测试集
    :param input_dir: 输入图像目录
    :param output_dir: 输出主目录
    """
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录 {input_dir} 不存在")
        return
    
    # 创建输出子目录
    occlusion_dir = os.path.join(output_dir, "Test-Occlusion")
    blur_dir = os.path.join(output_dir, "Test-Blur")
    light_dir = os.path.join(output_dir, "Test-Light")
    
    os.makedirs(occlusion_dir, exist_ok=True)
    os.makedirs(blur_dir, exist_ok=True)
    os.makedirs(light_dir, exist_ok=True)
    
    # 获取输入目录中的所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(input_dir) 
                  if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"错误：输入目录 {input_dir} 中没有找到图像文件")
        return
    
    print(f"开始处理 {len(image_files)} 张图像...")
    
    # 处理每张图像
    for i, image_file in enumerate(image_files):
        input_path = os.path.join(input_dir, image_file)
        file_name, file_ext = os.path.splitext(image_file)
        
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                print(f"警告：无法读取图像 {image_file}")
                continue
            
            # 1. 生成遮挡测试集图像
            occlusion_image = apply_occlusion(image)
            occlusion_output_path = os.path.join(occlusion_dir, image_file)
            cv2.imwrite(occlusion_output_path, occlusion_image)
            
            # 2. 生成模糊测试集图像
            blur_image = apply_motion_blur(image)
            blur_output_path = os.path.join(blur_dir, image_file)
            cv2.imwrite(blur_output_path, blur_image)
            
            # 3. 生成光照测试集图像
            light_image = apply_extreme_illumination(image)
            light_output_path = os.path.join(light_dir, image_file)
            cv2.imwrite(light_output_path, light_image)
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"已处理 {i + 1}/{len(image_files)} 张图像")
                
        except Exception as e:
            print(f"处理图像 {image_file} 时出错：{str(e)}")
    
    print(f"处理完成！生成的测试集保存在：")
    print(f"- 遮挡测试集：{occlusion_dir}")
    print(f"- 模糊测试集：{blur_dir}")
    print(f"- 光照测试集：{light_dir}")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 默认输入目录：ultralytics-main\dataset\dataset_tomato leaf\images\test
    DEFAULT_INPUT_DIR = r"e:\ultralytics-20240707\ultralytics-main\dataset\dataset_tomato leaf\images\test"
    
    # 默认输出目录：test_image_output
    DEFAULT_OUTPUT_DIR = "test_image_output"
    
    # 运行测试集生成
    generate_test_sets(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)
