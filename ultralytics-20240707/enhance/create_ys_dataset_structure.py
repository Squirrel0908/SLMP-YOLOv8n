import os

# 创建YS_dataset_test目录结构
base_dir = 'e:\\ultralytics-20240707\\YS_dataset_test'

# 需要创建的子目录
subdirs = ['test_weather', 'test_occlusion', 'test_blur', 'test_illumination']

for subdir in subdirs:
    # 创建子目录
    os.makedirs(os.path.join(base_dir, subdir), exist_ok=True)
    # 在每个子目录下创建image和label文件夹
    os.makedirs(os.path.join(base_dir, subdir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, subdir, 'label'), exist_ok=True)

print(f"目录结构已创建在 {base_dir}")