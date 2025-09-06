# RIFE 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'RIFE/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/ECCV2022-RIFE')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/ECCV2022-RIFE')

# you can add your custom imports here
from model.RIFE import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'Model':
                # Create interface object
                self.model_obj = Model(**kwargs)
                self.result.interface_return = self.model_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.model_obj = Model(**kwargs)
                self.result.interface_return = self.model_obj

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
            if dispatch_key == 'train':
                self.result.interface_return = self.model_obj.train(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'eval':
                self.result.interface_return = self.model_obj.eval(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'device':
                self.result.interface_return = self.model_obj.device(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load_model':
                self.result.interface_return = self.model_obj.load_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'save_model':
                self.result.interface_return = self.model_obj.save_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference':
                self.result.interface_return = self.model_obj.inference(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'update':
                self.result.interface_return = self.model_obj.update(**kwargs)
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
