# BasicSR 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'BasicSR/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/BasicSR')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/BasicSR')

# you can add your custom imports here
from basicsr.archs.basicvsr_arch import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'BasicVSR':
                # Create interface object for BasicVSR
                self.basic_vsr_obj = BasicVSR(**kwargs)
                self.result.interface_return = self.basic_vsr_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.basic_vsr_obj = BasicVSR(**kwargs)
                self.result.interface_return = self.basic_vsr_obj

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
            if dispatch_key == 'get_flow':
                # Call the get_flow method from BasicVSR
                self.result.interface_return = self.basic_vsr_obj.get_flow(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'forward':
                # Call the forward method from BasicVSR
                self.result.interface_return = self.basic_vsr_obj.forward(**kwargs)
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
