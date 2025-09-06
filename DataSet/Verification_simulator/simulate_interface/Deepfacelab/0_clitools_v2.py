from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Deepfacelab import *
exe = Executor('Deepfacelab', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/DeepFaceLab/main.py'
import multiprocessing
from core.leras import nn
import os
import sys
import time
import argparse
from core import pathex
from core import osex
from pathlib import Path
from core.interact import interact as io
from mainscripts import FacesetEnhancer
from mainscripts import VideoEd
if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 6):
    raise Exception('This program requires at least Python 3.6')
nn.initialize_main_env()
exit_code = 0
input_dir = RESOURCES_PATH + 'images/test_images_floder'
cpu_only = False
force_gpu_idxs = None

def process_faceset_enhancer():
    osex.set_process_lowest_prio()
    exe.run('process_folder', dirpath=input_dir, cpu_only=cpu_only, force_gpu_idxs=force_gpu_idxs)
input_file = RESOURCES_PATH + 'videos/test_video.mp4'
output_dir = FILE_RECORD_PATH
output_ext = 'png'
fps = 30

def process_videoed_extract_video():
    osex.set_process_lowest_prio()
    exe.run('extract_video', input_file=input_file, output_dir=output_dir, output_ext=output_ext, fps=fps)

def process_videoed_cut_video():
    osex.set_process_lowest_prio()
    exe.run('cut_video', input_file=input_file, from_time='00:00:00.000', to_time='00:01:00.000', audio_track_id=None, bitrate=None)

def process_videoed_denoise_image_sequence():
    osex.set_process_lowest_prio()
    exe.run('denoise_image_sequence', input_dir=input_dir, factor=5)

def process_videoed_video_from_sequence():
    osex.set_process_lowest_prio()
    exe.run('video_from_sequence', input_dir=input_dir, output_file=FILE_RECORD_PATH + '/video.mp4', reference_file=None, ext='png', fps=fps, bitrate=None, include_audio=False, lossless=False)
process_faceset_enhancer()
process_videoed_extract_video()
process_videoed_cut_video()
process_videoed_denoise_image_sequence()
process_videoed_video_from_sequence()
if exit_code == 0:
    print('Done.')
exit(exit_code)