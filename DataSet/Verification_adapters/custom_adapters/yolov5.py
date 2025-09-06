# yolov5 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'yolov5/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/yolov5')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/yolov5')

# you can add your custom imports here
from models.common import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'DetectMultiBackend':
                # Create interface object
                self.detect_multi_backend_obj = DetectMultiBackend(**kwargs)
                self.result.interface_return = self.detect_multi_backend_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.detect_multi_backend_obj = DetectMultiBackend(**kwargs)
                self.result.interface_return = self.detect_multi_backend_obj

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
            if dispatch_key == 'forward':
                # Call the forward method of the DetectMultiBackend instance
                self.result.interface_return = self.detect_multi_backend_obj.forward(**kwargs)
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
