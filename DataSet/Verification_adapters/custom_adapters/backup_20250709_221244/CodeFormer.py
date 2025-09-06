# CodeFormer 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'CodeFormer/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/CodeFormer')

# 可以在此位置后添加导包部分代码
import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite
from basicsr.utils import img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
import warnings
from basicsr.utils.video_util import VideoReader
from basicsr.utils.video_util import VideoWriter


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'FaceRestoreHelper':
                # 创建接口对象
                self.face_restore_helper = FaceRestoreHelper(**kwargs)
                self.result.interface_return = self.face_restore_helper
            else:
                pass
            
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 创建接口对象失败: {e}")

    def run(self, name: str, **kwargs):
        try:
            if name == 'set_upscale_factor':
                self.result.interface_return = self.face_restore_helper.set_upscale_factor(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'read_image':
                self.result.interface_return = self.face_restore_helper.read_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'init_dlib':
                self.result.interface_return = self.face_restore_helper.init_dlib(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'get_face_landmarks_5_dlib':
                self.result.interface_return = self.face_restore_helper.get_face_landmarks_5_dlib(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'get_face_landmarks_5':
                self.result.interface_return = self.face_restore_helper.get_face_landmarks_5(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'align_warp_face':
                self.result.interface_return = self.face_restore_helper.align_warp_face(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'get_inverse_affine':
                self.result.interface_return = self.face_restore_helper.get_inverse_affine(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'add_restored_face':
                self.result.interface_return = self.face_restore_helper.add_restored_face(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'paste_faces_to_input_image':
                self.result.interface_return = self.face_restore_helper.paste_faces_to_input_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'clean_all':
                self.result.interface_return = self.face_restore_helper.clean_all(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            else:
                raise ValueError(f"Unknown method name: {name}")

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行方法 {name} 失败: {e}")

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('FaceRestoreHelper')
adapter_additional_data['functions'].append('set_upscale_factor')
adapter_additional_data['functions'].append('read_image')
adapter_additional_data['functions'].append('init_dlib')
adapter_additional_data['functions'].append('get_face_landmarks_5_dlib')
adapter_additional_data['functions'].append('get_face_landmarks_5')
adapter_additional_data['functions'].append('align_warp_face')
adapter_additional_data['functions'].append('get_inverse_affine')
adapter_additional_data['functions'].append('add_restored_face')
adapter_additional_data['functions'].append('paste_faces_to_input_image')
adapter_additional_data['functions'].append('clean_all')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
