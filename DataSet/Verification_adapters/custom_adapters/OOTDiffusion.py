# OOTDiffusion 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'OOTDiffusion/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/OOTDiffusion')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/OOTDiffusion/run')

# you can add your custom imports here
from ootd.inference_ootd_hd import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'OOTDiffusionHD':
                self.oot_diffusion_hd_obj = OOTDiffusionHD(**kwargs)
                self.result.interface_return = self.oot_diffusion_hd_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.oot_diffusion_hd_obj = OOTDiffusionHD(**kwargs)
                self.result.interface_return = self.oot_diffusion_hd_obj

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
            if dispatch_key == 'call':
                # Call the __call__ method of OOTDiffusionHD
                self.result.interface_return = self.oot_diffusion_hd_obj(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'tokenize_captions':
                # Call the tokenize_captions method of OOTDiffusionHD
                self.result.interface_return = self.oot_diffusion_hd_obj.tokenize_captions(**kwargs)
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
