import warnings
warnings.filterwarnings('ignore')
import argparse, yaml, copy
from ultralytics.models.yolo.detect.compress import DetectionCompressor, DetectionFinetune
from ultralytics.models.yolo.segment.compress import SegmentationCompressor, SegmentationFinetune
from ultralytics.models.yolo.pose.compress import PoseCompressor, PoseFinetune
from ultralytics.models.yolo.obb.compress import OBBCompressor, OBBFinetune

def compress(param_dict):
    with open(param_dict['sl_hyp'], errors='ignore') as f:
        sl_hyp = yaml.safe_load(f)
    param_dict.update(sl_hyp)
    param_dict['name'] = f'{param_dict["name"]}-prune'
    param_dict['patience'] = 0
    compressor = DetectionCompressor(overrides=param_dict)
    # compressor = SegmentationCompressor(overrides=param_dict)
    # compressor = PoseCompressor(overrides=param_dict)
    # compressor = OBBCompressor(overrides=param_dict)
    prune_model_path = compressor.compress()
    return prune_model_path

def finetune(param_dict, prune_model_path):
    param_dict['model'] = prune_model_path
    param_dict['name'] = f'{param_dict["name"]}-finetune'
    trainer = DetectionFinetune(overrides=param_dict)
    # trainer = SegmentationFinetune(overrides=param_dict)
    # trainer = PoseFinetune(overrides=param_dict)
    # trainer = OBBFinetune(overrides=param_dict)
    trainer.train()

if __name__ == '__main__':
    param_dict = {
        # origin
        'model': r'/hy-tmp/runs/train/slc-yolov8n/weights/best.pt',
        'data':r'/hy-tmp/ultralytics-20240707/src/dataset/data.yaml',
        'imgsz': 640,
        'epochs': 200,
        'batch': 64, #仅剪枝建议64+2组合；稀疏训练建议32+2组合
        'workers': 2,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 10,
        'project':'/hy-tmp/runs/prune/exp3-0.005-base-train',
        'name':'slc-yolov8n-groupsl-0.005-speed2.0-base-train',
        'save_period': 20,
        # prune
        'prune_method':'group_sl',
        'global_pruning': True,
        'speed_up': 2.0,
        'reg': 0.05,
        'sl_epochs': 500,
        'sl_hyp': r'/hy-tmp/ultralytics-20240707/src/ultralytics/cfg/hyp.scratch.sl.yaml',
        'sl_model': r'/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-prune/weights/best.pt',
        # 'sl_model': None,
    }
    
    prune_model_path = compress(copy.deepcopy(param_dict))
    finetune(copy.deepcopy(param_dict), prune_model_path)