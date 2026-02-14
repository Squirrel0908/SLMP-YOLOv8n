# 弃用
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt')
    model.val(data=r'/hy-tmp/ultralytics-20240707/src/dataset/data.yaml',
              split='test',
              imgsz=640,
              batch=64,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='/hy-tmp/runs/test/slcp-yolov8n',
              name='slc-yolov8n-0.01-2-weather',
              )