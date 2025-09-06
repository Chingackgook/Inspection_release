# deep_live_cam 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'deep_live_cam/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Deep-Live-Cam')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Deep-Live-Cam')

# 可以在此位置后添加导包部分代码
from modules.processors.frame.face_swapper import *


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
from modules.utilities import has_image_extension
from modules.utilities import is_image
from modules.utilities import is_video
from modules.utilities import detect_fps
from modules.utilities import create_video
from modules.utilities import extract_frames
from modules.utilities import get_temp_frame_paths
from modules.utilities import restore_audio
from modules.utilities import create_temp
from modules.utilities import move_temp
from modules.utilities import clean_temp
from modules.utilities import normalize_output_path
import ctypes
import resource


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # 不需要初始化任何接口类的对象
        pass

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'pre_check':
                self.result.interface_return = pre_check(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'pre_start':
                self.result.interface_return = pre_start(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_face_swapper':
                self.result.interface_return = get_face_swapper(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'swap_face':
                self.result.interface_return = swap_face(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process_frame':
                self.result.interface_return = process_frame(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process_frame_v2':
                self.result.interface_return = process_frame_v2(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process_frames':
                self.result.interface_return = process_frames(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process_image':
                self.result.interface_return = process_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process_video':
                self.result.interface_return = process_video(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'create_lower_mouth_mask':
                self.result.interface_return = create_lower_mouth_mask(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'draw_mouth_mask_visualization':
                self.result.interface_return = draw_mouth_mask_visualization(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'apply_mouth_area':
                self.result.interface_return = apply_mouth_area(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'create_face_mask':
                self.result.interface_return = create_face_mask(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'apply_color_transfer':
                self.result.interface_return = apply_color_transfer(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行接口 {dispatch_key} 失败: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
