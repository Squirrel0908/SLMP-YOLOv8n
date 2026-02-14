import torch
print(torch.cuda.is_available())  # 应该是 True
print(torch.cuda.device_count())  # 应该是 1 或更多
print(torch.cuda.current_device())  # 应该是 0
import torch
print(torch.__version__)
# from PIL import Image
# print("PIL loaded successfully!")
# import requests
# print("Requests imported successfully!")
# print("Version:", requests.__version__)