# Omost 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Omost/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Omost')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Omost')
# add
os.environ['HF_HOME'] = os.path.join(os.path.dirname('/mnt/autor_name/haoTingDeWenJianJia/Omost/gradio_app.py'), 'hf_download')
# end add 只是为了我们方便测试，不加这个完全不影响
# you can add your custom imports here
from lib_omost.pipeline import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Interface object for StableDiffusionXLOmostPipeline
        try:
            if interface_class_name == 'StableDiffusionXLOmostPipeline':
                # Create interface object
                self.class1_obj = StableDiffusionXLOmostPipeline(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = StableDiffusionXLOmostPipeline(**kwargs)
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
            if dispatch_key == '__call__':
                # Call the __call__ method of StableDiffusionXLOmostPipeline
                self.result.interface_return = self.class1_obj(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'encode_bag_of_subprompts_greedy':
                self.result.interface_return = self.class1_obj.encode_bag_of_subprompts_greedy(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'all_conds_from_canvas':
                self.result.interface_return = self.class1_obj.all_conds_from_canvas(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'encode_cropped_prompt_77tokens':
                self.result.interface_return = self.class1_obj.encode_cropped_prompt_77tokens(**kwargs)
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
