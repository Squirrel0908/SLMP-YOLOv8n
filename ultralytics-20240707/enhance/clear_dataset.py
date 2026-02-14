import os
import shutil

# 设置要删除文件的根目录
root_dir = r"e:\ultralytics-20240707\YS_dataset_test"

# 遍历四个子文件夹
subdirs = ['test_weather', 'test_occlusion', 'test_blur', 'test_illumination']

for subdir in subdirs:
    # 删除image目录下的所有文件
    image_dir = os.path.join(root_dir, subdir, 'image')
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"已删除{image_dir}下的所有文件")
    
    # 删除label目录下的所有文件
    label_dir = os.path.join(root_dir, subdir, 'label')
    if os.path.exists(label_dir):
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"已删除{label_dir}下的所有文件")

print("所有文件已删除完成！")
