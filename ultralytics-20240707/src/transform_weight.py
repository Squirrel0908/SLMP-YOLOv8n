import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect.c2f_transfer import replace_c2f_with_c2f_v2, replace_c2f_v2_with_c2f

if __name__ == '__main__':
    input = torch.randn((1, 3, 640, 640))
    
    # C2f -> C2f_V2
    # model_weight_path = 'runs/train/yolov8-repvit-RepNCSPELAN/weights/best.pt'
    # model = YOLO(model_weight_path)
    # model.model.eval()
    # pre_res = model.model(input)[0]
    # replace_c2f_with_c2f_v2(model.model)
    # model.model.eval()
    # after_res = model.model(input)[0]
    # print(torch.mean(pre_res - after_res))
    # torch.save({'model':model.model.half()}, f'{model_weight_path[:model_weight_path.rfind(".")]}_v2.pt')
    
    # C2f_V2 -> C2f
    model_v2_weight_path = 'runs/prune/yolov8n-light-lamp-exp1-finetune/weights/best.pt'
    model = YOLO(model_v2_weight_path)
    model.model.eval()
    pre_res = model.model(input)[0]
    replace_c2f_v2_with_c2f(model.model)
    model.model.eval()
    after_res = model.model(input)[0]
    print(torch.mean(pre_res - after_res))
    torch.save({'model':model.model.half()}, f'{model_v2_weight_path[:model_v2_weight_path.rfind(".")]}_notv2.pt')