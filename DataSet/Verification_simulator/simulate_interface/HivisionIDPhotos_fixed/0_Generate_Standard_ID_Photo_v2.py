from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.HivisionIDPhotos_fixed import *
exe = Executor('HivisionIDPhotos_fixed', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# Import the existing package
import os
import cv2
import argparse
import numpy as np
from hivision.error import FaceError
from hivision.utils import hex_to_rgb
from hivision.utils import resize_image_to_kb
from hivision.utils import add_background
from hivision.utils import save_image_dpi_to_bytes
from hivision import IDCreator
from hivision.creator.layout_calculator import generate_layout_array
from hivision.creator.layout_calculator import generate_layout_image
from hivision.creator.choose_handler import choose_handler
from hivision.utils import hex_to_rgb
from hivision.utils import resize_image_to_kb
# end

import os
import cv2
import numpy as np
from hivision.error import FaceError
from hivision.utils import hex_to_rgb, save_image_dpi_to_bytes

input_image_dir = RESOURCES_PATH + 'images/test_image.jpg'

# Parts that may need manual modification:
output_image_dir = os.path.join(FILE_RECORD_PATH, 'image.jpg')  # Updated to use FILE_RECORD_PATH
# end

# Parts that may need manual modification:
height = 413
width = 295
color = '638cce'
hd = True
render = 0
dpi = 300
face_align = False
matting_model = 'modnet_photographic_portrait_matting'
face_detect_model = 'mtcnn'
inference_type = 'idphoto'
creator = exe.create_interface_objects(interface_class_name='IDCreator')

input_image = cv2.imread(input_image_dir, cv2.IMREAD_UNCHANGED)
if inference_type == 'idphoto':
    size = (int(height), int(width))
    try:
        result = exe.run('call', image=input_image, size=size, face_alignment=face_align)
    except FaceError:
        print('人脸数量不等于 1，请上传单张人脸的图像。')
    else:
        save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), output_image_dir, dpi=dpi)
        (file_name, file_extension) = os.path.splitext(output_image_dir)
        new_file_name = file_name + '_hd' + file_extension
        save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=dpi)