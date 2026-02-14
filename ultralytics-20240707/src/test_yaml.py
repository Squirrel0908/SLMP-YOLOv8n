import os
from ultralytics import YOLO
from tqdm import tqdm

if __name__ == '__main__':
    error_result = []
    
    # 1. 自动获取当前脚本所在的目录，并拼接出 v8 配置文件夹的路径
    # 假设你现在就在 ultralytics-main 目录下运行
    base_cfg_path = 'ultralytics-20240707/src/ultralytics/cfg/models/v8'
    
    if not os.path.exists(base_cfg_path):
        print(f"❌ 路径不存在: {os.path.abspath(base_cfg_path)}")
        exit()

    # 2. 遍历文件夹
    yaml_files = os.listdir(base_cfg_path)
    
    for yaml_name in tqdm(yaml_files):
        # 只处理 .yaml 文件，并排除 rtdetr 和 cls
        if yaml_name.endswith('.yaml') and 'rtdetr' not in yaml_name and 'cls' not in yaml_name:
            # 拼接完整的文件路径
            full_path = os.path.join(base_cfg_path, yaml_name)
            
            try:
                # 尝试加载模型
                model = YOLO(full_path)
                # print(f"✅ 成功加载: {yaml_name}")
                
                # 如果你想测试 CDDA 注册是否成功，这里不需要 model.profile
                # model.info() 就足够了
                
            except Exception as e:
                error_result.append(f'{yaml_name}: {e}')

    # 3. 打印结果
    if error_result:
        print("\n--- 报错统计 ---")
        for i in error_result:
            print(i)
    else:
        print("\n✅ 所有 YAML 配置文件加载正常！")