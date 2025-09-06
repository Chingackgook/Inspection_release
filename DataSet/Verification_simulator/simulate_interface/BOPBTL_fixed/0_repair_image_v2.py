from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.BOPBTL_fixed import *
exe = Executor('BOPBTL_fixed', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2

def data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = (ow, oh)
    if scale:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256
    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)
    return img if h == ph and w == pw else img.resize((w, h), method)

def data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Resize((256, 256), Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)

def irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype('uint8')
    mask_np = np.array(mask).astype('uint8')
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255
    return Image.fromarray(img_new.astype('uint8')).convert('RGB')

def parameter_set(opt):
    opt.serial_batches = True
    opt.no_flip = True
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = './checkpoints/restoration'
    opt.test_input = './input'
    opt.test_mask = './mask'
    # 可能需要手动修改的部分：
    if opt.Quality_restore:
        opt.name = 'mapping_quality'
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, 'VAE_A_quality')
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, 'VAE_B_quality')
    if opt.Scratch_and_Quality_restore:
        opt.NL_res = True
        opt.use_SN = True
        opt.correlation_renormalize = True
        opt.NL_use_mask = True
        opt.NL_fusion_method = 'combine'
        opt.non_local = 'Setting_42'
        opt.name = 'mapping_scratch'
        opt.load_pretrainA = os.path.join(opt.checkpoints_dir, 'VAE_A_quality')
        opt.load_pretrainB = os.path.join(opt.checkpoints_dir, 'VAE_B_scratch')
        if opt.HR:
            opt.mapping_exp = 1
            opt.inference_optimize = True
            opt.mask_dilation = 3
            opt.name = 'mapping_Patch_Attention'
    # end

opt = TestOptions().parse(save=False)
parameter_set(opt)
model = exe.create_interface_objects(interface_class_name='Pix2PixHDModel_Mapping')
_ = exe.run('initialize', opt=opt)
model.eval()

# 创建输出目录
for path in ['input_image', 'restored_image', 'origin']:
    os.makedirs(os.path.join(FILE_RECORD_PATH, path), exist_ok=True)

input_loader = sorted(os.listdir(opt.test_input))
dataset_size = len(input_loader)
mask_loader = sorted(os.listdir(opt.test_mask)) if opt.test_mask != '' else []
img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
mask_transform = transforms.ToTensor()

for i in range(dataset_size):
    input_name = input_loader[i]
    input_file = RESOURCES_PATH + 'images/test_image.jpg'
    if not os.path.isfile(input_file):
        print(f'Skipping non-file {input_name}')
        continue
    input = Image.open(input_file).convert('RGB')
    print(f'Now you are processing {input_name}')
    
    if opt.NL_use_mask:
        mask_name = mask_loader[i]
        mask = RESOURCES_PATH + 'images/test_image.jpg'
        if opt.mask_dilation != 0:
            kernel = np.ones((3, 3), np.uint8)
            mask = np.array(mask)
            mask = cv2.dilate(mask, kernel, iterations=opt.mask_dilation)
            mask = Image.fromarray(mask.astype('uint8'))
        origin = RESOURCES_PATH + 'images/test_image.png'
        input = irregular_hole_synthesize(input, mask)
        mask = mask_transform(mask)[:1, :, :].unsqueeze(0)
        input = img_transform(input).unsqueeze(0)
    else:
        if opt.test_mode == 'Scale':
            input = data_transforms(input, scale=True)
        elif opt.test_mode == 'Full':
            input = data_transforms(input, scale=False)
        elif opt.test_mode == 'Crop':
            input = data_transforms_rgb_old(input)
        origin = input
        input = img_transform(input).unsqueeze(0)
        mask = torch.zeros_like(input)

    try:
        with torch.no_grad():
            generated = exe.run('inference', label=input, inst=mask)
    except Exception as ex:
        print(f'Skip {input_name} due to an error:\n{str(ex)}')
        continue

    output_name = RESOURCES_PATH + 'images/test_image.png'
    vutils.save_image((input + 1.0) / 2.0, os.path.join(FILE_RECORD_PATH, 'input_image', output_name), nrow=1, padding=0, normalize=True)
    vutils.save_image((generated.data.cpu() + 1.0) / 2.0, os.path.join(FILE_RECORD_PATH, 'restored_image', output_name), nrow=1, padding=0, normalize=True)
    origin.save(os.path.join(FILE_RECORD_PATH, 'origin', output_name))