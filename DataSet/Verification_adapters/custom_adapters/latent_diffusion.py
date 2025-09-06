# latent_diffusion 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'latent_diffusion/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/latent-diffusion')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/latent-diffusion')

# you can add your custom imports here
from ldm.models.diffusion.ddim import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.ddim_sampler_obj = None  # Interface object for DDIMSampler
        try:
            if interface_class_name == 'DDIMSampler':
                # Create interface object
                self.ddim_sampler_obj = DDIMSampler(**kwargs)
                self.result.interface_return = self.ddim_sampler_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.ddim_sampler_obj = DDIMSampler(**kwargs)
                self.result.interface_return = self.ddim_sampler_obj

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
            if dispatch_key == 'sample':
                # Call sample method from DDIMSampler
                self.result.interface_return = self.ddim_sampler_obj.sample(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ddim_sampling':
                # Call ddim_sampling method from DDIMSampler
                self.result.interface_return = self.ddim_sampler_obj.ddim_sampling(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'p_sample_ddim':
                # Call p_sample_ddim method from DDIMSampler
                self.result.interface_return = self.ddim_sampler_obj.p_sample_ddim(**kwargs)
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
