# DragGAN 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'DragGAN/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/DragGAN')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/DragGAN')

# you can add your custom imports here
from gen_images import *

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
            if dispatch_key == 'parse_range':
                self.result.interface_return = parse_range(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'parse_vec2':
                self.result.interface_return = parse_vec2(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'make_transform':
                self.result.interface_return = make_transform(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate_images':
                generate_images.callback(**kwargs)  # This function does not return a value
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
                self.result.interface_return = None  # No return value from generate_images
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
