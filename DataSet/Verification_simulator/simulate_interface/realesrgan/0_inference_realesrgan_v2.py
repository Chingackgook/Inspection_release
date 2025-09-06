from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.realesrgan import *
exe = Executor('realesrgan','simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package
import argparse
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from gfpgan import GFPGANer
# end

def run_real_esrgan():
    """Inference demo for Real-ESRGAN with hardcoded parameters.
    """
    input_path = 'inputs'
    
    # Parts that may need manual modification:
    model_name = 'RealESRGAN_x4plus'  # Model name
    output_path = os.path.join(FILE_RECORD_PATH, 'results')  # Output path
    denoise_strength = 0.5  # Denoise strength
    outscale = 4  # Output scale
    model_path = None
    suffix = 'out'
    tile = 0
    tile_pad = 10
    pre_pad = 0
    face_enhance = False
    fp32 = False
    alpha_upsampler = 'realesrgan'
    ext = 'auto'
    gpu_id = None
    # end

    model_name = model_name.split('.')[0]
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth', 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
    
    if model_path is not None:
        model_path = model_path
    else:
        model_path = os.path.join('weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    
    dni_weight = None
    if model_name == 'realesr-general-x4v3' and denoise_strength != 1:
        wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
        model_path = [model_path, wdn_model_path]
        dni_weight = [denoise_strength, 1 - denoise_strength]
    
    upsampler = exe.create_interface_objects(interface_class_name='RealESRGANer', scale=netscale, model_path=model_path, dni_weight=dni_weight, model=model, tile=tile, tile_pad=tile_pad, pre_pad=pre_pad, half=not fp32, gpu_id=gpu_id)
    
    if face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = exe.create_interface_objects(interface_class_name='GFPGANer', model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=outscale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
    
    os.makedirs(output_path, exist_ok=True)
    
    if os.path.isfile(input_path):
        paths = [input_path]
    else:
        paths = sorted(glob.glob(os.path.join(input_path, '*')))
    
    for (idx, path) in enumerate(paths):
        (imgname, extension) = os.path.splitext(os.path.basename(path))
        print('Testing', idx, imgname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f'Warning: Failed to read image {path}')
            continue
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
            raise ValueError(f'Invalid image dimensions: {img.shape}')
        
        try:
            if face_enhance:
                (_, _, output) = exe.run('enhance', img=img, outscale=None, alpha_upsampler='realesrgan')
            else:
                (output, _) = exe.run('enhance', img=img, outscale=outscale, alpha_upsampler='realesrgan')
        except RuntimeError as error:
            print('Error', error)
            print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
        else:
            if ext == 'auto':
                extension = extension[1:]
            else:
                extension = ext
            if img_mode == 'RGBA':
                extension = 'png'
            if suffix == '':
                save_path = os.path.join(output_path, f'{imgname}.{extension}')
            else:
                save_path = os.path.join(output_path, f'{imgname}_{suffix}.{extension}')
            cv2.imwrite(save_path, output)

run_real_esrgan()