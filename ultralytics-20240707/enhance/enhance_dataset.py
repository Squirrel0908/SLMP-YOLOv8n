import os
import cv2
import shutil
import albumentations as A
import random

# 输入和输出目录（使用相对路径避免中文字符编码问题）
input_image_dir = 'ultralytics-main/dataset/dataset_tomato_leaf/image/train'
input_label_dir = 'ultralytics-main/dataset/dataset_tomato_leaf/label/train'
output_image_dir = 'YS_dataset_prune_train/image/train'
output_label_dir = 'YS_dataset_prune_train/label/train'

# 确保输出目录存在
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# 定义天气增强变换（保持原有的参数设置）
weather_transform = A.Compose([
    A.OneOf([
        # 暴雨模式：不仅变暗，还有粗雨线
        A.RandomRain(
            brightness_coefficient=0.8, # 更暗
            drop_width=1,               # 雨滴宽度
            blur_value=5,               # 雨线模糊度，调大一点更像水汽
            rain_type='heavy',          # 暴雨
            p=1.0
        ),
        # 浓雾模式：保持当前的设置
        A.RandomFog(
            fog_coef_lower=0.3, 
            fog_coef_upper=0.8, # 上限提高，模拟极其浓的雾
            alpha_coef=0.1, 
            p=1.0
        )
    ], p=1.0)
])

# 定义模糊增强变换
blur_transform = A.Compose([
    A.OneOf([
        A.Blur(blur_limit=10, p=1.0),
        A.GaussianBlur(blur_limit=10, p=1.0),
        A.MotionBlur(blur_limit=15, p=1.0)
    ], p=1.0)
])

# 定义光照增强变换
illumination_transform = A.Compose([
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.3, 0.3), p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0)
    ], p=1.0)
])

# 定义遮挡增强变换
occlusion_transform = A.Compose([
    A.OneOf([
        A.CoarseDropout(max_holes=10, max_height=30, max_width=30, min_holes=5, min_height=10, min_width=10, fill_value=0, mask_fill_value=None, p=1.0),
        A.GridDropout(ratio=0.2, p=1.0),
        A.RandomShadow(num_shadows_lower=1, num_shadows_upper=3, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1.0)
    ], p=1.0)
])

# 获取所有图片文件
image_files = [f for f in os.listdir(input_image_dir) if f.endswith('.jpg')]
image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 按数字顺序排序

# 所有增强变换列表
transforms = [
    ('weather', weather_transform),
    ('blur', blur_transform),
    ('illumination', illumination_transform),
    ('occlusion', occlusion_transform)
]

# 计算图片数量
total_images = len(image_files)

print(f"总共有 {total_images} 张图片")
print(f"将对所有 {total_images} 张图片进行增强")
print(f"每张图片随机选择一种增强效果")
print(f"最终总共有 {total_images * 2} 张图像")

# 遍历所有图片
for i, image_file in enumerate(image_files):
    # 获取图片路径和名称
    image_path = os.path.join(input_image_dir, image_file)
    image_name, image_ext = os.path.splitext(image_file)
    
    # 获取对应的标签文件路径
    label_file = image_name + '.txt'
    label_path = os.path.join(input_label_dir, label_file)
    
    # 检查标签文件是否存在
    if not os.path.exists(label_path):
        print(f"警告：图片 {image_file} 对应的标签文件 {label_file} 不存在，跳过处理")
        continue
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"警告：无法读取图片 {image_file}，跳过处理")
        continue
    
    # 复制原始图像和标签到输出目录
    output_original_image_path = os.path.join(output_image_dir, image_file)
    output_original_label_path = os.path.join(output_label_dir, label_file)
    shutil.copy2(image_path, output_original_image_path)
    shutil.copy2(label_path, output_original_label_path)
    
    # 随机选择一种增强方式
    transform_name, selected_transform = random.choice(transforms)
    
    # 应用增强
    augmented = selected_transform(image=image)
    augmented_image = augmented['image']
    
    # 保存增强后的图像和标签
    augmented_image_file = f"{image_name}_{transform_name}{image_ext}"
    augmented_label_file = f"{image_name}_{transform_name}.txt"
    output_augmented_image_path = os.path.join(output_image_dir, augmented_image_file)
    output_augmented_label_path = os.path.join(output_label_dir, augmented_label_file)
    cv2.imwrite(output_augmented_image_path, augmented_image)
    shutil.copy2(label_path, output_augmented_label_path)
    
    # 显示进度
    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1} 张图片")

# 验证输出
final_image_count = len(os.listdir(output_image_dir))
final_label_count = len(os.listdir(output_label_dir))

print(f"处理完成！")
print(f"最终输出目录包含 {final_image_count} 张图像")
print(f"最终输出目录包含 {final_label_count} 个标签文件")
print(f"预期的图像数量：{total_images * 2}")
print(f"预期的标签文件数量：{total_images * 2}")

if final_image_count == total_images * 2 and final_label_count == total_images * 2:
    print("输出结果符合预期！")
else:
    print("输出结果与预期不符，请检查是否有错误发生。")
