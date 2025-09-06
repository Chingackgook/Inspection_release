# CodeFormer 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'CodeFormer/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/CodeFormer')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/CodeFormer')

# you can add your custom imports here
from facelib.utils.face_restoration_helper import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def __init__(self):
        super().__init__()
        self.face_restore_helper = None  # Initialize the FaceRestoreHelper object

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'FaceRestoreHelper':
                # Create interface object for FaceRestoreHelper
                self.face_restore_helper = FaceRestoreHelper(**kwargs)
                self.result.interface_return = self.face_restore_helper
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.face_restore_helper = FaceRestoreHelper(**kwargs)
                self.result.interface_return = self.face_restore_helper

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'set_upscale_factor':
                self.result.interface_return = self.face_restore_helper.set_upscale_factor(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'read_image':
                self.result.interface_return = self.face_restore_helper.read_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'init_dlib':
                self.result.interface_return = self.face_restore_helper.init_dlib(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_face_landmarks_5_dlib':
                self.result.interface_return = self.face_restore_helper.get_face_landmarks_5_dlib(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_face_landmarks_5':
                self.result.interface_return = self.face_restore_helper.get_face_landmarks_5(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'align_warp_face':
                self.result.interface_return = self.face_restore_helper.align_warp_face(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_inverse_affine':
                self.result.interface_return = self.face_restore_helper.get_inverse_affine(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'add_restored_face':
                self.result.interface_return = self.face_restore_helper.add_restored_face(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'paste_faces_to_input_image':
                self.result.interface_return = self.face_restore_helper.paste_faces_to_input_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'clean_all':
                self.result.interface_return = self.face_restore_helper.clean_all(**kwargs)
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
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
