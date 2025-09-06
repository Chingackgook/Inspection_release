# SadTalker 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'SadTalker/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/SadTalker')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/SadTalker')

# you can add your custom imports here
from src.test_audio2coeff import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Interface object for Audio2Coeff
        try:
            if interface_class_name == 'Audio2Coeff':
                # Create interface object
                self.class1_obj = Audio2Coeff(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = Audio2Coeff(**kwargs)
                self.result.interface_return = self.class1_obj

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
            if dispatch_key == 'load_cpk':
                # Call top-level function
                self.result.interface_return = load_cpk(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate':
                # Call method from Audio2Coeff
                self.result.interface_return = self.class1_obj.generate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'using_refpose':
                # Call method from Audio2Coeff
                self.result.interface_return = self.class1_obj.using_refpose(**kwargs)
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
