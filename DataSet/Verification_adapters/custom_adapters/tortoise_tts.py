# tortoise_tts 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'tortoise_tts/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/tortoise-tts/tortoise')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/tortoise-tts/tortoise')

# you can add your custom imports here
from api import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Interface object for TextToSpeech
        try:
            if interface_class_name == 'TextToSpeech':
                # Create interface object
                self.class1_obj = TextToSpeech(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = TextToSpeech(**kwargs)
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
            if dispatch_key == 'tts_with_preset':
                # Call tts_with_preset method
                self.result.interface_return = self.class1_obj.tts_with_preset(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'tts':
                # Call tts method
                self.result.interface_return = self.class1_obj.tts(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_conditioning_latents':
                # Call get_conditioning_latents method
                self.result.interface_return = self.class1_obj.get_conditioning_latents(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_random_conditioning_latents':
                # Call get_random_conditioning_latents method
                self.result.interface_return = self.class1_obj.get_random_conditioning_latents()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'deterministic_state':
                # Call deterministic_state method
                self.result.interface_return = self.class1_obj.deterministic_state(**kwargs)
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
