import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.realesrgan import ENV_DIR
from Inspection.adapters.custom_adapters.realesrgan import *
exe = Executor('realesrgan', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import cv2
import glob
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

input_image_path = os.path.join(ENV_DIR, 'limingyang.jpg')
output_folder = os.path.join(FILE_RECORD_PATH, 'results')
model_name = 'RealESRGAN_x4plus'
model_path = None  # 可以根据需要设置
tile = 0
tile_pad = 10
pre_pad = 0
outscale = 4
face_enhance = False
gpu_id = 1

# 确定模型
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
netscale = 4
file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']

# 确定模型路径
if model_path is None:
    model_path = os.path.join(ENV_DIR, 'weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ENV_DIR, 'weights'), progress=True, file_name=None)

# 初始化 RealESRGANer
exe.create_interface_objects(
    scale=netscale,
    model_path=model_path,
    model=model,
    tile=tile,
    tile_pad=tile_pad,
    pre_pad=pre_pad,
    half=False,
    gpu_id=gpu_id
)

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 读取图像
img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

# 处理图像通道
if img is None:
    print(f"Warning: Failed to read image {input_image_path}")
else:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img_mode = 'RGB'
    elif len(img.shape) == 3:
        if img.shape[2] == 4:
            img_mode = 'RGBA'
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_mode = 'RGB'
    else:
        raise ValueError(f"Invalid image dimensions: {img.shape}")

    # 预处理图像
    exe.run("pre_process", img=img)

    # 增强图像
    output_img, img_mode = exe.run("enhance", img=img, outscale=outscale)

    # 后处理图像
    output_img = exe.run("post_process", output_img=output_img)

    # 保存输出图像
    extension = 'jpg'  # 可以根据需要设置
    save_path = os.path.join(output_folder, f'output_image.{extension}')
    cv2.imwrite(save_path, output_img)
