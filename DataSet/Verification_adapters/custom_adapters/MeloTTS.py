# MeloTTS 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'MeloTTS/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/MeloTTS')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/MeloTTS')

# you can add your custom imports here
from melo.api import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'TTS':
                # Create interface object for TTS
                self.tts_obj = TTS(**kwargs)
                self.result.interface_return = self.tts_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.tts_obj = TTS(**kwargs)
                self.result.interface_return = self.tts_obj

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
            if dispatch_key == 'tts_to_file':
                # Call tts_to_file method from TTS
                self.result.interface_return = self.tts_obj.tts_to_file(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'audio_numpy_concat':
                # Call audio_numpy_concat static method from TTS
                self.result.interface_return = TTS.audio_numpy_concat(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'split_sentences_into_pieces':
                # Call split_sentences_into_pieces static method from TTS
                self.result.interface_return = TTS.split_sentences_into_pieces(**kwargs)
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
