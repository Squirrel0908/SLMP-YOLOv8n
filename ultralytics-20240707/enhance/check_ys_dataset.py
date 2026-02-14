import os

# 输出目录
base_dir = 'e:\\ultralytics-20240707\\YS_dataset_test'

# 需要检查的子目录
subdirs = ['test_weather', 'test_occlusion', 'test_blur', 'test_illumination']

print("检查YS_dataset_test目录中每个子目录的文件数量：")
print("=" * 50)

for subdir in subdirs:
    print(f"\n{subdir}:")
    
    # 检查图片数量
    image_dir = os.path.join(base_dir, subdir, 'image')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"  图片数量: {len(image_files)}")
    
    # 检查标签数量
    label_dir = os.path.join(base_dir, subdir, 'label')
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    print(f"  标签数量: {len(label_files)}")
    
    # 检查是否一一对应
    if len(image_files) == len(label_files):
        print("  ✅ 图片和标签数量一致")
    else:
        print("  ❌ 图片和标签数量不一致")

print("\n" + "=" * 50)
print("检查完成")