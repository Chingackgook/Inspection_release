# first_order_model 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'first_order_model/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/first-order-model')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/first-order-model')

# you can add your custom imports here
from modules.generator import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Interface object for OcclusionAwareGenerator
        try:
            if interface_class_name == 'OcclusionAwareGenerator':
                # Create interface object
                self.class1_obj = OcclusionAwareGenerator(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = OcclusionAwareGenerator(**kwargs)
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
            if dispatch_key == 'forward':
                # Call the forward method from OcclusionAwareGenerator
                self.result.interface_return = self.class1_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'deform_input':
                # Call the deform_input method from OcclusionAwareGenerator
                self.result.interface_return = self.class1_obj.deform_input(**kwargs)
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
