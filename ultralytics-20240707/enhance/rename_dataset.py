import os
import shutil

# 数据集批量重命名python脚本

# 源数据集路径
SOURCE_BASE = r"e:\ultralytics-20240707\ultralytics-main\dataset\dataset_tomato_leaf"

# 目标数据集路径
TARGET_BASE = r"e:\ultralytics-20240707\sort_dataset"

# 数据集分割（train, val, test）
SPLITS = ["train", "val", "test"]

# 支持的图像扩展名
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

def main():
    # 创建目标目录结构
    for split in SPLITS:
        # 创建图像目标目录
        os.makedirs(os.path.join(TARGET_BASE, "image", split), exist_ok=True)
        # 创建标签目标目录
        os.makedirs(os.path.join(TARGET_BASE, "label", split), exist_ok=True)
    
    # 处理每个数据集分割
    for split in SPLITS:
        print(f"\n处理 {split} 数据集...")
        
        # 源目录路径
        image_source_dir = os.path.join(SOURCE_BASE, "images", split)
        label_source_dir = os.path.join(SOURCE_BASE, "labels", split)
        
        # 获取所有图像文件
        image_files = []
        for file in os.listdir(image_source_dir):
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_files.append(file)
        
        # 按文件名排序（可选，保持一致性）
        image_files.sort()
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 处理每张图像及其对应的标签
        for idx, image_file in enumerate(image_files, 1):
            # 获取图像文件名（不含扩展名）和扩展名
            image_name, image_ext = os.path.splitext(image_file)
            
            # 对应的标签文件名
            label_file = image_name + ".txt"
            
            # 源文件路径
            image_source_path = os.path.join(image_source_dir, image_file)
            label_source_path = os.path.join(label_source_dir, label_file)
            
            # 检查标签文件是否存在
            if not os.path.exists(label_source_path):
                print(f"警告：图像 {image_file} 对应的标签文件 {label_file} 不存在")
                continue
            
            # 新的文件名（数字序号）
            new_name = str(idx)
            
            # 目标文件路径
            image_target_path = os.path.join(TARGET_BASE, "image", split, f"{new_name}{image_ext}")
            label_target_path = os.path.join(TARGET_BASE, "label", split, f"{new_name}.txt")
            
            # 复制文件
            shutil.copy2(image_source_path, image_target_path)
            shutil.copy2(label_source_path, label_target_path)
            
            # 显示进度
            if idx % 100 == 0:
                print(f"  已处理 {idx}/{len(image_files)} 个样本")
        
        print(f"{split} 数据集处理完成")

if __name__ == "__main__":
    print("开始重命名数据集...")
    print(f"源路径: {SOURCE_BASE}")
    print(f"目标路径: {TARGET_BASE}")
    main()
    print("\n数据集重命名完成！")
