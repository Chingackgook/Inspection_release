# AnimateDiff 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'AnimateDiff/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/AnimateDiff')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/AnimateDiff')

# you can add your custom imports here
from animatediff.pipelines.pipeline_animation import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # This will hold the AnimationPipeline object
        try:
            if interface_class_name == 'AnimationPipeline':
                # Create interface object for AnimationPipeline
                self.class1_obj = AnimationPipeline(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = AnimationPipeline(**kwargs)
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
            if dispatch_key == 'enable_vae_slicing':
                self.result.interface_return = self.class1_obj.enable_vae_slicing(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'disable_vae_slicing':
                self.result.interface_return = self.class1_obj.disable_vae_slicing(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'enable_sequential_cpu_offload':
                self.result.interface_return = self.class1_obj.enable_sequential_cpu_offload(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'decode_latents':
                self.result.interface_return = self.class1_obj.decode_latents(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'prepare_extra_step_kwargs':
                self.result.interface_return = self.class1_obj.prepare_extra_step_kwargs(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'check_inputs':
                self.result.interface_return = self.class1_obj.check_inputs(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'prepare_latents':
                self.result.interface_return = self.class1_obj.prepare_latents(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == '__call__':
                self.result.interface_return = self.class1_obj.__call__(**kwargs)
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
