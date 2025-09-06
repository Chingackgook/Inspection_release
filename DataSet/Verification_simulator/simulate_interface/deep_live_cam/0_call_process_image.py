import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.deep_live_cam import ENV_DIR
from Inspection.adapters.custom_adapters.deep_live_cam import *
exe = Executor('deep_live_cam', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
import sys
import warnings
import shutil
import platform
import signal
import torch
import onnxruntime
import tensorflow

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

modules.globals.source_path = '/test_source.jpg'
modules.globals.target_path = '/test_target.jpg'
modules.globals.output_path = 'output_dir'
modules.globals.frame_processors = ['face_swapper']
modules.globals.headless = True
modules.globals.keep_fps = True
modules.globals.keep_audio = True
modules.globals.keep_frames = False
modules.globals.many_faces = False
modules.globals.mouth_mask = False
modules.globals.nsfw_filter = False
modules.globals.map_faces = False
modules.globals.video_encoder = 'libx264'
modules.globals.video_quality = 18
modules.globals.live_mirror = False
modules.globals.live_resizable = False
modules.globals.max_memory = 8
modules.globals.execution_providers = ['CUDAExecutionProvider']
modules.globals.execution_threads = 1
modules.globals.lang = 'zh-cn'

# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
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

def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if not modules.globals.headless:
        ui.update_status(message)

def start() -> None:

    update_status('Processing...')
    
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, FILE_RECORD_PATH)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            spath = ENV_DIR + modules.globals.source_path
            tpath = ENV_DIR + modules.globals.target_path
            exe.run("process_image", source_path=spath, target_path=tpath, output_path=FILE_RECORD_PATH)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    if not modules.globals.map_faces:
        update_status('Creating temp resources...')
        create_temp(ENV_DIR + modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(ENV_DIR + modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(ENV_DIR + modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        exe.run("process_video", source_path=ENV_DIR + modules.globals.source_path, temp_frame_paths=temp_frame_paths)
        release_resources()
    
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(ENV_DIR + modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(FILE_RECORD_PATH, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(FILE_RECORD_PATH)
    
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(ENV_DIR + modules.globals.target_path, FILE_RECORD_PATH)
    else:
        move_temp(ENV_DIR + modules.globals.target_path, FILE_RECORD_PATH)
    
    # clean and validate
    clean_temp(ENV_DIR + modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')

def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(ENV_DIR + modules.globals.target_path)
    if to_quit: quit()

def run() -> None:
    if not exe.run("pre_check"):
        return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()

# 运行程序
run()
