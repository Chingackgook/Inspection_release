# audiocraft 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'audiocraft/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/audiocraft')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/audiocraft')

# you can add your custom imports here
from audiocraft.models.musicgen import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'MusicGen':
                # Create interface object for MusicGen
                self.class1_obj = MusicGen.get_pretrained(**kwargs)
                # originally, it was like this:
                # self.class1_obj = MusicGen(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = MusicGen.get_pretrained(**kwargs)
                # originally, it was like this:
                # self.class1_obj = MusicGen(**kwargs)
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
            if dispatch_key == 'get_pretrained':
                # Call static method from MusicGen
                self.result.interface_return = MusicGen.get_pretrained(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'set_generation_params':
                # Call instance method from MusicGen
                self.result.interface_return = self.class1_obj.set_generation_params(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'set_style_conditioner_params':
                # Call instance method from MusicGen
                self.result.interface_return = self.class1_obj.set_style_conditioner_params(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate_with_chroma':
                # Call instance method from MusicGen
                self.result.interface_return = self.class1_obj.generate_with_chroma(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate':
                # Call instance method from MusicGen
                self.result.interface_return = self.class1_obj.generate(**kwargs)
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
