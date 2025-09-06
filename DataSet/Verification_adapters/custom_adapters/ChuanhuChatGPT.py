# ChuanhuChatGPT 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'ChuanhuChatGPT/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/ChuanhuChatGPT')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/ChuanhuChatGPT')

# you can add your custom imports here
from modules.models.OpenAIVision import *
from modules.models.base_model import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Placeholder for OpenAIVisionClient instance
        try:
            if interface_class_name == 'OpenAIVisionClient':
                # Create interface object
                self.class1_obj = OpenAIVisionClient(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = OpenAIVisionClient(**kwargs)
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
            if dispatch_key == 'get_answer_stream_iter':
                self.result.interface_return = self.class1_obj.get_answer_stream_iter()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_answer_at_once':
                self.result.interface_return = self.class1_obj.get_answer_at_once()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'count_token':
                self.result.interface_return = self.class1_obj.count_token(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'count_image_tokens':
                self.result.interface_return = self.class1_obj.count_image_tokens(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'billing_info':
                self.result.interface_return = self.class1_obj.billing_info()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'set_key':
                self.result.interface_return = self.class1_obj.set_key(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'auto_name_chat_history':
                self.result.interface_return = self.class1_obj.auto_name_chat_history(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                self.result.interface_return = self.class1_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'retry':
                self.result.interface_return = self.class1_obj.retry(**kwargs)
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
