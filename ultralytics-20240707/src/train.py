import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'/hy-tmp/ultralytics-20240707/src/ultralytics/cfg/models/v8/SLC-yolov8n.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/hy-tmp/ultralytics-20240707/src/dataset/data.yaml',
                cache=True,
                imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=10,
                workers=8,
                device='0',
                optimizer='SGD', # using SGD
                patience=0, # close earlystop
                # resume='', # last.pt path
                amp=True, # close amp
                # fraction=0.2,
                project='/hy-tmp/runs/train',
                name='slc-yolov8n',
                )      