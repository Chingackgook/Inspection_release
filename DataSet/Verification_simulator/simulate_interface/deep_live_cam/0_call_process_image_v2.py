from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.deep_live_cam import *
exe = Executor('deep_live_cam', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
import sys
import warnings
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
import ctypes
import resource
import os
import sys
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from typing import List
import platform
import signal
import shutil
import torch
import onnxruntime
import tensorflow
import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path
if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
modules.globals.source_path = RESOURCES_PATH + 'images/test_image.jpg'
modules.globals.target_path = RESOURCES_PATH + 'images/test_image.jpg'
modules.globals.output_path = os.path.join(FILE_RECORD_PATH, 'output_image.jpg')
modules.globals.frame_processors = ['face_swapper']
modules.globals.keep_fps = False
modules.globals.keep_audio = True
modules.globals.keep_frames = False
modules.globals.many_faces = False
modules.globals.mouth_mask = False
modules.globals.nsfw_filter = False
modules.globals.map_faces = False
modules.globals.video_encoder = 'libx264'
modules.globals.video_quality = 18
modules.globals.lang = 'en'
modules.globals.max_memory = 16
modules.globals.execution_providers = ['cpu']
modules.globals.execution_threads = 8
modules.globals.headless = True

def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]

def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers())) if any((execution_provider in encoded_execution_provider for execution_provider in execution_providers))]

def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16

def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())

def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8

def limit_resources() -> None:
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))

def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()

def pre_check() -> bool:
    if sys.version_info < (3, 9):
        print('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        print('ffmpeg is not installed.')
        return False
    return True

def update_status(message: str, scope: str='DLC.CORE') -> None:
    print(f'[{scope}] {message}')

def start() -> None:
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status(message='Processing...')
    if has_image_extension(modules.globals.target_path):
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print('Error copying file:', str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status(message='Progressing...', scope=frame_processor.NAME)
            exe.run('process_image', source_path=modules.globals.source_path, target_path=modules.globals.output_path, output_path=modules.globals.output_path)
            release_resources()
        update_status(message='Processing to image succeed!')
        return
    if not modules.globals.map_faces:
        update_status(message='Creating temp resources...')
        create_temp(target_path=modules.globals.target_path)
        update_status(message='Extracting frames...')
        extract_frames(target_path=modules.globals.target_path)
    temp_frame_paths = get_temp_frame_paths(target_path=modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status(message='Progressing...', scope=frame_processor.NAME)
        exe.run('process_video', source_path=modules.globals.source_path, temp_frame_paths=temp_frame_paths)
        release_resources()
    if modules.globals.keep_fps:
        update_status(message='Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(message=f'Creating video with {fps} fps...')
        create_video(target_path=modules.globals.target_path, fps=fps)
    else:
        update_status(message='Creating video with 30.0 fps...')
        create_video(target_path=modules.globals.target_path)
    if modules.globals.keep_audio:
        update_status(message='Restoring audio...')
        restore_audio(target_path=modules.globals.target_path, output_path=modules.globals.output_path)
    else:
        move_temp(target_path=modules.globals.target_path, output_path=modules.globals.output_path)
    clean_temp(target_path=modules.globals.target_path)
    update_status(message='Processing to video succeed!')

def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(target_path=modules.globals.target_path)
    if to_quit:
        quit()

def run() -> None:
    if not exe.run('pre_check'):
        return
    limit_resources()
    start()
run()