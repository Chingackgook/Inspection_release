# consistency_models 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'consistency_models/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/consistency_models')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/consistency_models')

# you can add your custom imports here
from cm.karras_diffusion import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            # Since there are no specific interface classes to initialize, we can skip this part.
            # If needed, we can add logic to handle future interface classes.
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
            if dispatch_key == 'karras_sample':
                # Call the top-level function karras_sample
                self.result.interface_return = karras_sample(**kwargs)
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
