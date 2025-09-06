# CosyVoice 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'CosyVoice/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/CosyVoice')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/CosyVoice')

# you can add your custom imports here
from cosyvoice.cli.cosyvoice import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            self.cosy_voice_obj = None  # Interface object for CosyVoice2
            if interface_class_name == 'CosyVoice2':
                # Create interface object
                self.cosy_voice_obj = CosyVoice2(**kwargs)
                self.result.interface_return = self.cosy_voice_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.cosy_voice_obj = CosyVoice2(**kwargs)
                self.result.interface_return = self.cosy_voice_obj

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
            if dispatch_key == 'list_available_spks':
                self.result.interface_return = self.cosy_voice_obj.list_available_spks(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'add_zero_shot_spk':
                self.result.interface_return = self.cosy_voice_obj.add_zero_shot_spk(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'save_spkinfo':
                self.result.interface_return = self.cosy_voice_obj.save_spkinfo(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_sft':
                self.result.interface_return = self.cosy_voice_obj.inference_sft(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_zero_shot':
                self.result.interface_return = self.cosy_voice_obj.inference_zero_shot(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_cross_lingual':
                self.result.interface_return = self.cosy_voice_obj.inference_cross_lingual(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_instruct':
                self.result.interface_return = self.cosy_voice_obj.inference_instruct(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_vc':
                self.result.interface_return = self.cosy_voice_obj.inference_vc(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference_instruct2':
                self.result.interface_return = self.cosy_voice_obj.inference_instruct2(**kwargs)
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
