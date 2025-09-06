from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.SadTalker import *
exe = Executor('SadTalker','simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/SadTalker/interface.py'
# Import the existing package
from glob import glob
import shutil
import torch
from time import strftime
import os
import sys
import time
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.face3d.visualize import gen_composed_video
# end

def main(args):
    pic_path = args['source_image']
    audio_path = args['driven_audio']
    save_dir = os.path.join(args['result_dir'], strftime('%Y_%m_%d_%H.%M.%S'))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args['pose_style']
    device = args['device']
    batch_size = args['batch_size']
    input_yaw_list = args['input_yaw']
    input_pitch_list = args['input_pitch']
    input_roll_list = args['input_roll']
    ref_eyeblink = args['ref_eyeblink']
    ref_pose = args['ref_pose']
    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args['checkpoint_dir'], os.path.join(
        current_root_path, 'src/config'), args['size'], args['old_version'],
        args['preprocess'])
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio_to_coeff = exe.create_interface_objects(interface_class_name=
        'Audio2Coeff', sadtalker_path=sadtalker_paths, device=device)
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    
    # Replace output file root paths with FILE_RECORD_PATH
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, args['preprocess'], source_image_flag=
        True, pic_size=args['size'])
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return
    
    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(
            ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink,
            ref_eyeblink_frame_dir, args['preprocess'], source_image_flag=False
            )
    else:
        ref_eyeblink_coeff_path = None
    
    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[
                0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose,
                ref_pose_frame_dir, args['preprocess'], source_image_flag=False
                )
    else:
        ref_pose_coeff_path = None
    
    batch = get_data(first_coeff_path, audio_path, device,
        ref_eyeblink_coeff_path, still=args['still'])
    
    coeff_path = exe.run('generate', batch=batch, coeff_save_dir=FILE_RECORD_PATH,
        pose_style=pose_style, ref_pose_coeff_path=ref_pose_coeff_path)
    
    if args['face3dvis']:
        gen_composed_video(args, device, first_coeff_path, coeff_path,
            audio_path, os.path.join(FILE_RECORD_PATH, '3dface.mp4'))
    
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path,
        audio_path, batch_size, input_yaw_list, input_pitch_list,
        input_roll_list, expression_scale=args['expression_scale'],
        still_mode=args['still'], preprocess=args['preprocess'], size=args[
        'size'])
    
    result = animate_from_coeff.generate(data, save_dir, pic_path,
        crop_info, enhancer=args['enhancer'], background_enhancer=args[
        'background_enhancer'], preprocess=args['preprocess'], img_size=
        args['size'])
    
    shutil.move(result, os.path.join(FILE_RECORD_PATH, 'output_video.mp4'))
    print('The generated video is named:', os.path.join(FILE_RECORD_PATH, 'output_video.mp4'))
    
    if not args['verbose']:
        shutil.rmtree(save_dir)

args = {
    'driven_audio': './examples/driven_audio/bus_chinese.wav',
    'source_image': './examples/source_image/full_body_1.png',
    'ref_eyeblink': None,
    'ref_pose': None,
    'checkpoint_dir': './checkpoints',
    'result_dir': './results',
    'pose_style': 0,
    'batch_size': 2,
    'size': 256,
    'expression_scale': 1.0,
    'input_yaw': None,
    'input_pitch': None,
    'input_roll': None,
    'enhancer': None,
    'background_enhancer': None,
    'cpu': False,
    'face3dvis': False,
    'still': False,
    'preprocess': 'crop',
    'verbose': False,
    'old_version': False
}

# Determine device
if torch.cuda.is_available() and not args['cpu']:
    args['device'] = 'cuda'
else:
    args['device'] = 'cpu'

# Run the main logic directly
main(args)