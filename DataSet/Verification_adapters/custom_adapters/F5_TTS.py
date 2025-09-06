# F5_TTS 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'F5_TTS/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/F5-TTS/src')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/F5-TTS/src')

# you can add your custom imports here
from f5_tts.infer.utils_infer import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.result.fuc_name = 'create_interface_objects'
        try:
            if interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.result.interface_return = None  # No interface class to initialize
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.is_file = False
                self.result.file_path = ''
            else:
                # Handle specific interface class initialization if needed
                raise ValueError(f"Unknown interface class name: {interface_class_name}")

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")

    def run(self, dispatch_key: str, **kwargs):
        self.result.fuc_name = dispatch_key
        try:
            if dispatch_key == 'chunk_text':
                self.result.interface_return = chunk_text(**kwargs)
            elif dispatch_key == 'load_vocoder':
                self.result.interface_return = load_vocoder(**kwargs)
            elif dispatch_key == 'initialize_asr_pipeline':
                initialize_asr_pipeline(**kwargs)
                self.result.interface_return = None  # No return value
            elif dispatch_key == 'transcribe':
                self.result.interface_return = transcribe(**kwargs)
            elif dispatch_key == 'load_checkpoint':
                self.result.interface_return = load_checkpoint(**kwargs)
            elif dispatch_key == 'load_model':
                self.result.interface_return = load_model(**kwargs)
            elif dispatch_key == 'remove_silence_edges':
                self.result.interface_return = remove_silence_edges(**kwargs)
            elif dispatch_key == 'preprocess_ref_audio_text':
                self.result.interface_return = preprocess_ref_audio_text(**kwargs)
            elif dispatch_key == 'infer_process':
                self.result.interface_return = infer_process(**kwargs)
            elif dispatch_key == 'infer_batch_process':
                self.result.interface_return = infer_batch_process(**kwargs)
            elif dispatch_key == 'remove_silence_for_generated_wav':
                remove_silence_for_generated_wav(**kwargs)
                self.result.interface_return = None  # No return value
            elif dispatch_key == 'save_spectrogram':
                save_spectrogram(**kwargs)
                self.result.interface_return = None  # No return value
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

            self.result.is_success = True
            self.result.fail_reason = ''

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
