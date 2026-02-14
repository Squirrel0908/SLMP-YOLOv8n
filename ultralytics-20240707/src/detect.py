import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\ultralytics-20240707\ultralytics-main\yolov8n.pt') # select your model.pt path
    model.predict(source=r'E:\ultralytics-20240707\ultralytics-main\dataset\dataset_tomato leaf\images\test',
                  imgsz=640,
                  project='runs/detect',
                  name='yolov8n原始模型推理测试',
                  save=True,
                  save_txt=True
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True # visualize model features maps
                )