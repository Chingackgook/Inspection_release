# Vallex 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Vallex/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/VALL-E-X')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/VALL-E-X')

# you can add your custom imports here
from models.vallex import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'VALLE':
                # Create interface object for VALLE
                self.valle_obj = VALLE(**kwargs)
                self.result.interface_return = self.valle_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.valle_obj = VALLE(**kwargs)
                self.result.interface_return = self.valle_obj

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
            if dispatch_key == 'top_k_top_p_filtering':
                # Call top-level function
                self.result.interface_return = top_k_top_p_filtering(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'forward':
                # Call forward method from VALLE
                self.result.interface_return = self.valle_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference':
                # Call inference method from VALLE
                self.result.interface_return = self.valle_obj.inference(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'continual':
                # Call continual method from VALLE
                self.result.interface_return = self.valle_obj.continual(**kwargs)
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
