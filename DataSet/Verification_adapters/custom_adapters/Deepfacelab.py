# Deepfacelab 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Deepfacelab/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/DeepFaceLab')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/DeepFaceLab')

# you can add your custom imports here
from mainscripts.FacesetEnhancer import *
from mainscripts.VideoEd import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # No interface class objects need to be initialized since all are top-level functions
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_file = False
        self.result.file_path = ''
        self.result.interface_return = None  # No interface object created

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'process_folder':
                self.result.interface_return = process_folder(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'extract_video':
                self.result.interface_return = extract_video(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'cut_video':
                self.result.interface_return = cut_video(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'denoise_image_sequence':
                self.result.interface_return = denoise_image_sequence(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'video_from_sequence':
                self.result.interface_return = video_from_sequence(**kwargs)
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
