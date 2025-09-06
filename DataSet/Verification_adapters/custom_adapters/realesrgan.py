# realesrgan 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'realesrgan/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Real-ESRGAN')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Real-ESRGAN')

# you can add your custom imports here
from realesrgan.utils import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.realesrganer = None  # Interface object for RealESRGANer
        try:
            if interface_class_name == 'RealESRGANer':
                # Create interface object
                self.realesrganer = RealESRGANer(**kwargs)
                self.result.interface_return = self.realesrganer
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.realesrganer = RealESRGANer(**kwargs)
                self.result.interface_return = self.realesrganer

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
            if dispatch_key == 'pre_process':
                self.result.interface_return = self.realesrganer.pre_process(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'process':
                self.result.interface_return = self.realesrganer.process()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'tile_process':
                self.result.interface_return = self.realesrganer.tile_process()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'post_process':
                self.result.interface_return = self.realesrganer.post_process()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'enhance':
                self.result.interface_return = self.realesrganer.enhance(**kwargs)
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
