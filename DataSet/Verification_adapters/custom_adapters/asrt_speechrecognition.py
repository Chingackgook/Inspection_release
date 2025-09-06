# asrt_speechrecognition 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'asrt_speechrecognition/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/ASRT_SpeechRecognition')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/ASRT_SpeechRecognition')

# you can add your custom imports here
from speech_model import *
from language_model3 import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'ModelSpeech':
                self.class1_obj = ModelSpeech(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'ModelLanguage':
                self.class2_obj = ModelLanguage(**kwargs)
                self.result.interface_return = self.class2_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.class1_obj = ModelSpeech(**kwargs)
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
            if dispatch_key == 'ModelSpeech_train_model':
                self.result.interface_return = self.class1_obj.train_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_load_model':
                self.result.interface_return = self.class1_obj.load_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_save_model':
                self.result.interface_return = self.class1_obj.save_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_evaluate_model':
                self.result.interface_return = self.class1_obj.evaluate_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_predict':
                self.result.interface_return = self.class1_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_recognize_speech':
                self.result.interface_return = self.class1_obj.recognize_speech(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelSpeech_recognize_speech_from_file':
                self.result.interface_return = self.class1_obj.recognize_speech_from_file(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelLanguage_load_model':
                self.result.interface_return = self.class2_obj.load_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelLanguage_pinyin_to_text':
                self.result.interface_return = self.class2_obj.pinyin_to_text(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ModelLanguage_pinyin_stream_decode':
                self.result.interface_return = self.class2_obj.pinyin_stream_decode(**kwargs)
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
