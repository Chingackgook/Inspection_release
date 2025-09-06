# fish_speech_fixed 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'fish_speech_fixed/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/fish-speech')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/fish-speech')

# you can add your custom imports here
from fish_speech.models.text2semantic.inference import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Placeholder for any specific class object if needed
        self.class2_obj = None  # Placeholder for any specific class object if needed
        
        try:
            if interface_class_name == 'GenerateResponse':
                # Create interface object for GenerateResponse if needed
                self.class1_obj = GenerateResponse()
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'WrappedGenerateResponse':
                # Create interface object for WrappedGenerateResponse if needed
                self.class2_obj = WrappedGenerateResponse()
                self.result.interface_return = self.class2_obj
            elif interface_class_name == 'GenerateRequest':
                # Create interface object for GenerateRequest if needed
                self.class1_obj = GenerateRequest(request={}, response_queue=queue.Queue())
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.class1_obj = GenerateResponse()
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
            if dispatch_key == 'multinomial_sample_one_no_sync':
                self.result.interface_return = multinomial_sample_one_no_sync(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'logits_to_probs':
                self.result.interface_return = logits_to_probs(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'multinomial_sample_one_no_sync_agent':
                self.result.interface_return = multinomial_sample_one_no_sync_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'logits_to_probs_agent':
                self.result.interface_return = logits_to_probs_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'sample':
                self.result.interface_return = sample(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'sample_agent':
                self.result.interface_return = sample_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_one_token_ar_agent':
                self.result.interface_return = decode_one_token_ar_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_one_token_naive_agent':
                self.result.interface_return = decode_one_token_naive_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_one_token_ar':
                self.result.interface_return = decode_one_token_ar(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_one_token_naive':
                self.result.interface_return = decode_one_token_naive(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_n_tokens':
                self.result.interface_return = decode_n_tokens(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'generate':
                self.result.interface_return = generate(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'decode_n_tokens_agent':
                self.result.interface_return = decode_n_tokens_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'generate_agent':
                self.result.interface_return = generate_agent(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'encode_tokens':
                self.result.interface_return = encode_tokens(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'load_model':
                self.result.interface_return = load_model(**kwargs)
                self.result.is_success = True
            elif dispatch_key == 'generate_long':
                self.result.interface_return = generate_long(**kwargs)
                self.result.is_success = True
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

            self.result.fail_reason = ''
            self.result.fuc_name = dispatch_key

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
