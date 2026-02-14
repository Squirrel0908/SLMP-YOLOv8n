import sys
import os
import yaml
import warnings
import pandas as pd  # 使用 pandas 打印表格更美观
from tabulate import tabulate # 如果没有安装，下面有备选方案

# ==================== 0. 环境与路径设置 ====================
src_path = '/hy-tmp/ultralytics-20240707/src'
if src_path not in sys.path:
    sys.path.append(src_path)

warnings.filterwarnings('ignore')
from ultralytics import YOLO

# ==================== 1. 配置区域 ====================
# 模型权重路径
WEIGHT_PATH = '/hy-tmp/runs/train/slm-yolov8n/weights/best.pt'

# 结果保存根目录
RESULT_BASE_DIR = '/hy-tmp/Result/prune-test-result/slm-yolov8n'
os.makedirs(RESULT_BASE_DIR, exist_ok=True)

# 测试任务配置
TEST_TASKS = {
    'original':     '/hy-tmp/ultralytics-20240707/YS_dataset_test/test',
    'blur':         '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_blur',
    'illumination': '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_illumination',
    'occlusion':    '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_occlusion',
    'weather':      '/hy-tmp/ultralytics-20240707/YS_dataset_test/test_weather'
}

NC = 9
NAMES = ['Early Blight', 'Healthy', 'Late Blight', 'Leaf Miner', 'Leaf Mold', 
         'Mosaic Virus', 'Septoria', 'Spider Mites', 'Yellow Leaf Curl Virus']

# ==================== 2. 核心逻辑 ====================

def print_beautiful_table(records):
    """打印漂亮的终端表格"""
    df = pd.DataFrame(records)
    
    # 尝试计算相对于 original 的下降幅度
    try:
        baseline = df[df['Task'] == 'original'].iloc[0]
        # 计算 mAP50 的保持率 (Retention Rate)
        df['Retention(%)'] = (df['mAP50'] / baseline['mAP50'] * 100).round(2)
        df['Drop(%)'] = (100 - df['Retention(%)']).round(2)
    except:
        pass # 如果没有 original 任务，跳过计算

    print("\n" + "="*80)
    print(f"📊 鲁棒性测试汇总报告 | 模型: {os.path.basename(WEIGHT_PATH)}")
    print("="*80)
    
    # 使用 Pandas 的 to_markdown 或者直接打印
    # 如果环境没有安装 tabulate，pandas 会默认输出简单的 string 格式
    try:
        print(df.to_markdown(index=False, numalign="left", stralign="left"))
    except ImportError:
        print(df.to_string(index=False))
        
    print("="*80 + "\n")
    return df

def run_auto_val():
    model = YOLO(WEIGHT_PATH)
    print(f"✅ 成功加载模型: {WEIGHT_PATH}")
    
    # 用于存储所有任务的指标
    summary_records = []

    for task_name, task_path in TEST_TASKS.items():
        print(f"\n🚀 [正在执行] 任务场景: {task_name} ...")
        
        # 1. 生成临时 YAML
        tmp_yaml_data = {
            'path': task_path,
            'train': 'images',
            'val': 'images',
            'test': 'images',
            'nc': NC,
            'names': NAMES
        }
        
        tmp_yaml_path = f'tmp_val_{task_name}.yaml'
        with open(tmp_yaml_path, 'w') as f:
            yaml.dump(tmp_yaml_data, f)

        try:
            # 2. 执行验证
            results = model.val(
                data=tmp_yaml_path,
                split='test',
                imgsz=640,
                batch=64,
                device=0,
                project=RESULT_BASE_DIR,
                name=task_name,
                exist_ok=True,
                save_json=True,
                verbose=False # 关闭刷屏，保持清爽，结果会在最后汇总
            )
            
            # 3. 关键步骤：提取指标
            # results.box 包含：map50, map, mp, mr
            metrics = {
                'Task': task_name,
                'Precision': round(results.box.mp, 4),
                'Recall': round(results.box.mr, 4),
                'mAP50': round(results.box.map50, 4),
                'mAP50-95': round(results.box.map, 4)
            }
            summary_records.append(metrics)
            print(f"   └── ✅ 完成! mAP50: {metrics['mAP50']}")

        except Exception as e:
            print(f"   └── ❌ 失败: {str(e)}")
        
        finally:
            # 清理临时文件
            if os.path.exists(tmp_yaml_path):
                os.remove(tmp_yaml_path)

    # 4. 打印最终汇总表
    if summary_records:
        df = print_beautiful_table(summary_records)
        
        # 保存汇总 CSV 到结果目录，方便画图
        csv_path = os.path.join(RESULT_BASE_DIR, 'robustness_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"📁 汇总数据已保存至: {csv_path}")

if __name__ == '__main__':
    # 检查是否安装了 pandas 和 tabulate (为了美观)
    try:
        import pandas
    except ImportError:
        os.system('pip install pandas tabulate')
        
    run_auto_val()