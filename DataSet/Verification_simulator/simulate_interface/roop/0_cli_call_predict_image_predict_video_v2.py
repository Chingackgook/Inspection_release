from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.roop import *
exe = Executor('roop', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/roop/roop/core.py'
import os
import sys
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow
import roop.globals
import roop.metadata
import roop.ui as ui
from roop.predictor import predict_image
from roop.predictor import predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.utilities import has_image_extension
from roop.utilities import is_image
from roop.utilities import is_video
from roop.utilities import detect_fps
from roop.utilities import create_video
from roop.utilities import extract_frames
from roop.utilities import get_temp_frame_paths
from roop.utilities import restore_audio
from roop.utilities import create_temp
from roop.utilities import move_temp
from roop.utilities import clean_temp
from roop.utilities import normalize_output_path
import ctypes
import resource
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def parse_args() -> None:
    roop.globals.source_path = RESOURCES_PATH + 'images/test_image.jpg'
    roop.globals.target_path = RESOURCES_PATH + 'videos/test_video.mp4'
    roop.globals.output_path = FILE_RECORD_PATH
    roop.globals.frame_processors = ['face_swapper']
    roop.globals.keep_fps = True
    roop.globals.keep_frames = False
    roop.globals.skip_audio = False
    roop.globals.many_faces = False
    roop.globals.reference_face_position = 0
    roop.globals.reference_frame_number = 0
    roop.globals.similar_face_distance = 0.85
    roop.globals.temp_frame_format = 'png'
    roop.globals.temp_frame_quality = 0
    roop.globals.output_video_encoder = 'libx264'
    roop.globals.output_video_quality = 35
    roop.globals.max_memory = None
    roop.globals.execution_providers = ['cpu']
    roop.globals.execution_threads = 1
    roop.globals.headless = True

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for (provider, encoded_execution_provider) in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers())) if any((execution_provider in encoded_execution_provider for execution_provider in execution_providers))]

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8
    return 1

def limit_resources() -> None:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        print('[ROOP.CORE] Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('[ROOP.CORE] ffmpeg is not installed.')
        return False
    return True

def update_status(message: str, scope: str='ROOP.CORE') -> None:
    print(f'[{scope}] {message}')

def start() -> None:
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    if has_image_extension(roop.globals.target_path):
        if exe.run('predict_image', target_path=roop.globals.target_path):
            destroy()
        shutil.copy2(roop.globals.target_path, roop.globals.output_path)
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(roop.globals.source_path, roop.globals.output_path, roop.globals.output_path)
            frame_processor.post_process()
        if is_image(roop.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    if exe.run('predict_video', target_path=roop.globals.target_path):
        destroy()
    update_status('Creating temporary resources...')
    create_temp(roop.globals.target_path)
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Extracting frames with {fps} FPS...')
        extract_frames(roop.globals.target_path, fps)
    else:
        update_status('Extracting frames with 30 FPS...')
        extract_frames(roop.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(roop.globals.target_path)
    if temp_frame_paths:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.source_path, temp_frame_paths)
            frame_processor.post_process()
    else:
        update_status('Frames not found...')
        return
    if roop.globals.keep_fps:
        fps = detect_fps(roop.globals.target_path)
        update_status(f'Creating video with {fps} FPS...')
        create_video(roop.globals.target_path, fps)
    else:
        update_status('Creating video with 30 FPS...')
        create_video(roop.globals.target_path)
    if roop.globals.skip_audio:
        move_temp(roop.globals.target_path, roop.globals.output_path)
        update_status('Skipping audio...')
    else:
        if roop.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(roop.globals.target_path, roop.globals.output_path)
    update_status('Cleaning temporary resources...')
    clean_temp(roop.globals.target_path)
    if is_video(roop.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')

def destroy() -> None:
    if roop.globals.target_path:
        clean_temp(roop.globals.target_path)
    sys.exit()

def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    start()
run()