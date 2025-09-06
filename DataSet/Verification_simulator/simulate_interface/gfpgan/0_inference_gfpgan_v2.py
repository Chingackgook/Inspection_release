from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.gfpgan import *
exe = Executor('gfpgan','simulation')
FILE_RECORD_PATH = exe.now_record_path

import argparse
import cv2
import glob
import numpy as np
import os
import torch
from basicsr.utils import imwrite
from gfpgan import GFPGANer
import warnings
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def run_gfpgan(input_dir='inputs/whole_imgs', output_dir='results', version='1.3', upscale=2, bg_upsampler='realesrgan', bg_tile=400, suffix=None, only_center_face=False, aligned=False, ext='auto', weight=0.5):
    """Inference demo for GFPGAN (for users)."""
    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    if os.path.isfile(input_dir):
        img_list = [input_dir]
    else:
        img_list = sorted(glob.glob(os.path.join(input_dir, '*')))
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    if bg_upsampler == 'realesrgan':
        if not torch.cuda.is_available():
            warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. If you really want to use it, please modify the corresponding codes.')
            bg_upsampler = None
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            bg_upsampler = RealESRGANer(scale=2, model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', model=model, tile=bg_tile, tile_pad=10, pre_pad=0, half=True)
    else:
        bg_upsampler = None

    # Parts that may need manual modification:
    if version == '1':
        arch = 'original'
        channel_multiplier = 1
        model_name = 'GFPGANv1'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
    elif version == '1.2':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANCleanv1-NoCE-C2'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
    elif version == '1.3':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.3'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    elif version == '1.4':
        arch = 'clean'
        channel_multiplier = 2
        model_name = 'GFPGANv1.4'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
    elif version == 'RestoreFormer':
        arch = 'RestoreFormer'
        channel_multiplier = 2
        model_name = 'RestoreFormer'
        url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
    else:
        raise ValueError(f'Wrong model version {version}.')
    # end

    model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = os.path.join('gfpgan/weights', model_name + '.pth')
    if not os.path.isfile(model_path):
        model_path = url

    restorer = exe.create_interface_objects(interface_class_name='GFPGANer', model_path=model_path, upscale=upscale, arch=arch, channel_multiplier=channel_multiplier, bg_upsampler=bg_upsampler)

    for img_path in img_list:
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        # Run enhancement
        cropped_faces, restored_faces, restored_img = exe.run('enhance', img=input_img, has_aligned=aligned, only_center_face=only_center_face, paste_back=True, weight=weight)

        # Save cropped faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            save_crop_path = os.path.join(FILE_RECORD_PATH, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
            if suffix is not None:
                save_face_name = f'{basename}_{idx:02d}_{suffix}.png'
            else:
                save_face_name = f'{basename}_{idx:02d}.png'
            save_restore_path = os.path.join(FILE_RECORD_PATH, 'restored_faces', save_face_name)
            imwrite(restored_face, save_restore_path)
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(FILE_RECORD_PATH, 'cmp', f'{basename}_{idx:02d}.png'))

        if restored_img is not None:
            if ext == 'auto':
                extension = ext[1:]
            else:
                extension = ext
            if suffix is not None:
                save_restore_path = os.path.join(FILE_RECORD_PATH, 'restored_imgs', f'{basename}_{suffix}.{extension}')
            else:
                save_restore_path = os.path.join(FILE_RECORD_PATH, 'restored_imgs', f'{basename}.{extension}')
            imwrite(restored_img, save_restore_path)

    print(f'Results are in the [{FILE_RECORD_PATH}] folder.')

# Run the main logic directly
run_gfpgan()