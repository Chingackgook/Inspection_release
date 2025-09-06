from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.OOTDiffusion import *
exe = Executor('OOTDiffusion', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/OOTDiffusion/run/run_ootd.py'
from pathlib import Path
import sys
from PIL import Image
# add
from run.utils_ootd import get_mask_location
# end add
# origin code
# from utils_ootd import get_mask_location
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from ootd.inference_ootd_dc import OOTDiffusionDC
import argparse
gpu_id = 0
model_path = '/mnt/autor_name/Inspection/Env/OOTDiffusion/test_image.jpg' # add
cloth_path = '/mnt/autor_name/Inspection/Env/OOTDiffusion/test_image.jpg' # add
model_type = 'hd'
category = 0
image_scale = 2.0
n_steps = 20
n_samples = 4
seed = -1
PROJECT_ROOT = Path('/mnt/autor_name/haoTingDeWenJianJia/OOTDiffusion/run/run_ootd.py').absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
openpose_model = OpenPose(gpu_id)
parsing_model = Parsing(gpu_id)
category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']
if model_type == 'hd':
    model = exe.create_interface_objects(interface_class_name='OOTDiffusionHD', gpu_id=gpu_id)
elif model_type == 'dc':
    model = OOTDiffusionDC(gpu_id)
else:
    raise ValueError("model_type must be 'hd' or 'dc'!")
if model_type == 'hd' and category != 0:
    raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")
cloth_img = Image.open(cloth_path).resize((768, 1024))
model_img = Image.open(model_path).resize((768, 1024))
keypoints = openpose_model(model_img.resize((384, 512)))
model_parse, _ = parsing_model(model_img.resize((384, 512)))
mask, mask_gray = get_mask_location(model_type, category_dict_utils[category], model_parse, keypoints)
mask = mask.resize((768, 1024), Image.NEAREST)
mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)
masked_vton_img = Image.composite(mask_gray, model_img, mask)
masked_vton_img.save(f'{FILE_RECORD_PATH}/mask.jpg')
images = exe.run('call', model_type=model_type, category=category_dict[category], image_garm=cloth_img, image_vton=masked_vton_img, mask=mask, image_ori=model_img, num_samples=n_samples, num_steps=n_steps, image_scale=image_scale, seed=seed)
image_idx = 0
for image in images:
    image.save(f'{FILE_RECORD_PATH}/out_{model_type}_{image_idx}.png')
    image_idx += 1