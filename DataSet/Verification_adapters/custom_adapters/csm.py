# csm 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'csm/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/csm')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/csm')

# you can add your custom imports here
from generator import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.generator_obj = None  # Interface object for Generator
        try:
            if interface_class_name == 'Generator':
                # Create interface object
                self.generator_obj = load_csm_1b(**kwargs)  # Assuming kwargs contains device info
                self.result.interface_return = self.generator_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.generator_obj = load_csm_1b(**kwargs)
                self.result.interface_return = self.generator_obj

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
            if dispatch_key == 'load_llama3_tokenizer':
                # Call top-level function
                self.result.interface_return = load_llama3_tokenizer()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load_csm_1b':
                # Call top-level function
                self.result.interface_return = load_csm_1b(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate':
                # Call generate method from Generator
                if self.generator_obj is not None:
                    self.result.interface_return = self.generator_obj.generate(**kwargs)
                    self.result.is_success = True
                    self.result.fail_reason = ''
                    self.result.fuc_name = dispatch_key
                else:
                    raise ValueError("Generator object is not initialized.")
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
