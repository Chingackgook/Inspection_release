
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.HivisionIDPhotos_fixed import ENV_DIR
from Inspection.adapters.custom_adapters.HivisionIDPhotos_fixed import *
exe = Executor('HivisionIDPhotos_fixed','simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
import cv2
import numpy as np
from hivision.error import FaceError
from hivision.utils import save_image_dpi_to_bytes
from hivision import IDCreator
from hivision.creator.choose_handler import choose_handler

# 模拟输入参数
input_image_dir = os.path.join(ENV_DIR, "test0.jpg")  # 替换为实际的输入图像路径
output_image_dir = os.path.join(FILE_RECORD_PATH, "output/image.png")  # 替换为实际的输出图像路径
if not os.path.exists(os.path.dirname(output_image_dir)):
    os.makedirs(os.path.dirname(output_image_dir))
height = 413
width = 295
face_align = False
matting_model = "modnet_photographic_portrait_matting"
face_detect_model = "mtcnn"
dpi = 300

# ------------------- 选择抠图与人脸检测模型 -------------------
exe.create_interface_objects()
choose_handler(exe.adapter.id_creator, matting_model, face_detect_model)

# 读取输入图像
input_image = cv2.imread(input_image_dir, cv2.IMREAD_UNCHANGED)

# 如果模式是生成证件照
size = (height, width)
try:
    # 替换为 exe.run 调用
    result = exe.run("__call__", image=input_image, size=size, face_alignment=face_align)
except FaceError:
    print("人脸数量不等于 1，请上传单张人脸的图像。")
else:
    # 保存标准照
    save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), output_image_dir, dpi=dpi)

    # 保存高清照
    file_name, file_extension = os.path.splitext(output_image_dir)
    new_file_name = file_name + "_hd" + file_extension
    save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=dpi)
