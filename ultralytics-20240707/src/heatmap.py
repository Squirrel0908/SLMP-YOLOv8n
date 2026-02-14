import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil, sys
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy, non_max_suppression
from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            # Because of https://github.com/pytorch/pytorch/issues/61519,
            # we don't use backward hook to record gradients.
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output

        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return

        # Gradients are computed in reverse order
        def _store_grad(grad):
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def post_process(self, result):
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
  
    def __call__(self, x):
        self.gradients = []
        self.activations = []
        model_output = self.model(x)
        post_result, pre_post_boxes, post_boxes = self.post_process(model_output[0])
        return [[post_result, pre_post_boxes]]

    def release(self):
        for handle in self.handles:
            handle.remove()

class yolov8_target(torch.nn.Module):
    def __init__(self, ouput_type, conf, ratio) -> None:
        super().__init__()
        self.ouput_type = ouput_type
        self.conf = conf
        self.ratio = ratio
    
    def forward(self, data):
        post_result, pre_post_boxes = data
        result = []
        for i in trange(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf:
                break
            if self.ouput_type == 'class' or self.ouput_type == 'all':
                result.append(post_result[i].max())
            elif self.ouput_type == 'box' or self.ouput_type == 'all':
                for j in range(4):
                    result.append(pre_post_boxes[i, j])
        return sum(result)

class yolov8_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio, show_box, renormalize):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        model = attempt_load_weights(weight, device)
        model.info()
        for p in model.parameters():
            p.requires_grad_(True)
        model.eval()
        
        target = yolov8_target(backward_type, conf_threshold, ratio)
        target_layers = [model.model[l] for l in layer]
        method = eval(method)(model, target_layers, use_cuda=device.type == 'cuda')
        method.activations_and_grads = ActivationsAndGradients(model, target_layers, None)
        
        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int_)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        result = non_max_suppression(result, conf_thres=self.conf_threshold, iou_thres=0.65)[0]
        return result

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def renormalize_cam_in_bounding_boxes(self, boxes, image_float_np, grayscale_cam):
        """Normalize the CAM to be in the range [0, 1] 
        inside every bounding boxes, and zero outside of the bounding boxes. """
        renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(grayscale_cam.shape[1] - 1, x2), min(grayscale_cam.shape[0] - 1, y2)
            renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        renormalized_cam = scale_cam_image(renormalized_cam)
        eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
        return eigencam_image_renormalized
    
    def process(self, img_path, save_path):
        # 图像预处理
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img_float, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        
        # 1. 计算热力图
        try:
            grayscale_cam = self.method(tensor, [self.target])
        except Exception as e:
            print(f"Error at {img_path}: {e}")
            return
        
        grayscale_cam = grayscale_cam[0, :]
        
        # 2. 如果不需要框，直接合成纯净热力图
        if not self.show_box:
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
        else:
            # 如果要显示框，先处理预测结果
            pred = self.model(tensor)[0]
            # 关键修复：对类别/置信度部分进行 Sigmoid 处理
            # 注意：YOLOv8 的输出格式是 [batch, 4 + nc, anchors]
            # 我们只需要简单处理一下显示逻辑即可
            pred = self.post_process(pred) 
            
            cam_image = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            if self.renormalize:
                cam_image = self.renormalize_cam_in_bounding_boxes(pred[:, :4].cpu().detach().numpy().astype(np.int32), img_float, grayscale_cam)
            
            for data in pred:
                data = data.cpu().detach().numpy()
                # 修复 7.00 问题：这里获取真正的置信度
                conf = float(torch.sigmoid(torch.tensor(data[4:].max()))) 
                cam_image = self.draw_detections(data[:4], self.colors[int(data[4:].argmax())], f'{self.model_names[int(data[4:].argmax())]} {conf:.2f}', cam_image)
        
        # 保存结果
        cam_image = Image.fromarray(cam_image)
        cam_image.save(save_path)
        
    def __call__(self, img_path, save_path):
        # make dir if not exist
        os.makedirs(save_path, exist_ok=True)

        if os.path.isdir(img_path):
            for img_path_ in os.listdir(img_path):
                if img_path_.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.process(f'{img_path}/{img_path_}', f'{save_path}/{img_path_}')
        else:
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                self.process(img_path, f'{save_path}/result.png')
            else:
                print(f"Skipping non-PNG file: {img_path}")
        
# def get_params():
#     params = {
#         'weight': r'C:\Users\Sen\Desktop\ultralytics-20240707\runs\train\C2f_star_LADH_MLCA2\weights\best.pt', # 现在只需要指定权重即可,不需要指定cfg
#         'device': 'cuda:0',
#         'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM, EigenCAM, HiResCAM, LayerCAM, RandomCAM, EigenGradCAM
#         'layer': [10, 12, 14, 16, 18],
#         'backward_type': 'class', # class, box, all
#         'conf_threshold': 0.2, # 0.2
#         'ratio': 0.02, # 0.02-0.1
#         'show_box': False,
#         'renormalize': True
#     }
#     return params
def get_params(model_name):
    # 基础参数
    base_params = {
        'device': 'cuda:0',
        'method': 'GradCAMPlusPlus',
        'layer': [9],  #9 ，11
        'backward_type': 'class', 
        'conf_threshold': 0.2, 
        'ratio': 0.02, 
        'show_box': False,
        'renormalize': True
    }
    
    # 模型权重路径映射
    model_weights = {
        'slc': '/hy-tmp/runs/train/slc-yolov8n/weights/best.pt',
        'slcp': '/hy-tmp/runs/prune/exp3-0.005/slc-yolov8n-groupsl-0.005-finetune/weights/best.pt',
        'slm': '/hy-tmp/runs/train/slm-yolov8n/weights/best.pt',
        'baseline': '/hy-tmp/runs/train/yolov8n-baseline/weights/best.pt'
    }
    
    # 确保模型名称存在
    if model_name not in model_weights:
        raise ValueError(f"Model {model_name} not found. Available models: {list(model_weights.keys())}")
    
    # 更新权重路径
    params = base_params.copy()
    params['weight'] = model_weights[model_name]
    
    return params

def process_all_models(input_path, output_base):
    """
    处理所有模型的热力图生成
    
    Args:
        input_path: 输入图片目录路径
        output_base: 输出热力图基础目录路径
    """
    models = ['slc', 'slcp', 'slm', 'baseline']
    
    for model_name in models:
        print(f"Processing model: {model_name}")
        try:
            # 获取模型参数
            params = get_params(model_name)
            # 创建模型
            model = yolov8_heatmap(**params)
            # 生成输出路径
            output_path = os.path.join(output_base, model_name)
            # 处理输入路径（可以是单个文件或目录）
            model(input_path, output_path)
            print(f"Model {model_name} processed successfully!")
        except Exception as e:
            print(f"Error processing model {model_name}: {e}")

if __name__ == '__main__':
    # 输入图片路径（可以是单个图片文件或目录）
    input_path = r'/hy-tmp/Result/MLCA_CDDA_heatmap_diff/input'
    # 输出基础目录
    output_base = r'/hy-tmp/Result/MLCA_CDDA_heatmap_diff/output'
    
    # 处理所有模型
    process_all_models(input_path, output_base)