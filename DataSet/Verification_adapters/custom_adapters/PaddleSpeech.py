# PaddleSpeech 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'PaddleSpeech/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/PaddleSpeech')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/PaddleSpeech')

# you can add your custom imports here
from paddlespeech.cli.asr.infer import *
from paddlespeech.cli.text.infer import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'ASRExecutor':
                # Create ASRExecutor object
                self.asr_executor = ASRExecutor(**kwargs)
                self.result.interface_return = self.asr_executor
            elif interface_class_name == 'TextExecutor':
                # Create TextExecutor object
                self.text_executor = TextExecutor(**kwargs)
                self.result.interface_return = self.text_executor
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.asr_executor = ASRExecutor(**kwargs)
                self.result.interface_return = self.asr_executor

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
            if dispatch_key == 'ASRExecutor_execute':
                # Call execute method from ASRExecutor
                self.result.interface_return = self.asr_executor.execute(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'ASRExecutor___call__':
                # Call __call__ method from ASRExecutor
                self.result.interface_return = self.asr_executor(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'TextExecutor_execute':
                # Call execute method from TextExecutor
                self.result.interface_return = self.text_executor.execute(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'TextExecutor___call__':
                # Call __call__ method from TextExecutor
                self.result.interface_return = self.text_executor(**kwargs)
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
