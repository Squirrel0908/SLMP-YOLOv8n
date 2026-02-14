import os
import cv2
import numpy as np
import random

# 查看Test-Light目录中的一些图像
def verify_light_images():
    light_dir = os.path.join("test_image_output", "Test-Light")
    
    if not os.path.exists(light_dir):
        print(f"错误：光照测试集目录 {light_dir} 不存在")
        return
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(light_dir) 
                  if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print(f"错误：光照测试集目录 {light_dir} 中没有找到图像文件")
        return
    
    # 随机选择5张图像查看
    sample_images = random.sample(image_files, min(5, len(image_files)))
    
    print(f"查看 {len(sample_images)} 张随机选择的光照测试集图像：")
    
    for i, image_file in enumerate(sample_images):
        image_path = os.path.join(light_dir, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"警告：无法读取图像 {image_file}")
            continue
        
        # 检查图像是否有明显的黑色块
        # 计算黑色像素比例
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        black_pixels = np.sum(gray < 10)  # 像素值小于10视为黑色
        total_pixels = gray.size
        black_ratio = black_pixels / total_pixels
        
        print(f"图像 {image_file}：黑色像素比例 = {black_ratio:.4f} ({black_pixels}/{total_pixels})")
        
        # 显示图像信息
        print(f"  - 尺寸: {image.shape[0]}x{image.shape[1]}")
        
        # 如果黑色像素比例过高，打印警告
        if black_ratio > 0.1:  # 黑色像素超过10%视为异常
            print(f"  - 警告：黑色像素比例较高，可能存在问题")
        else:
            print(f"  - 黑色像素比例正常")

if __name__ == "__main__":
    verify_light_images()
