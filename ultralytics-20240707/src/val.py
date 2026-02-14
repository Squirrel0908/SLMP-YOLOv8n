import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'runs/env_test/yolov8n_baseline_best.pt')
    model.val(data='ultralytics-20240707/src/dataset/data.yaml',
              split='val',
              imgsz=640,
              batch=64,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='test1-yolov8n-baseline',
              )