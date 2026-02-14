import os

base_dir = 'e:\\ultralytics-20240707\\YS_dataset_test'

print("检查YS_dataset_test目录结构：")
print("=" * 50)

for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        print(f"\n{subdir}:")
        
        # 检查image目录
        image_dir = os.path.join(subdir_path, 'image')
        if os.path.exists(image_dir):
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            print(f"  图片: {len(image_files)}")
            if image_files:
                print(f"  示例图片: {image_files[:3]}...")
        else:
            print(f"  错误：缺少image目录")
        
        # 检查label目录
        label_dir = os.path.join(subdir_path, 'label')
        if os.path.exists(label_dir):
            label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
            print(f"  标签: {len(label_files)}")
            if label_files:
                print(f"  示例标签: {label_files[:3]}...")
        else:
            print(f"  错误：缺少label目录")

print("\n" + "=" * 50)
print("检查完成")
