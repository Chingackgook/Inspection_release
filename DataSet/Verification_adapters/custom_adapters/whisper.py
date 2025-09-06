# whisper 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'whisper/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/whisper')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/whisper')

# you can add your custom imports here
from whisper import *
from whisper.transcribe import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # No interface class objects to initialize, as there are only top-level functions.
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_file = False
        self.result.file_path = ''
        self.result.interface_return = None  # No interface object created

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'available_models':
                self.result.interface_return = available_models()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load_model':
                self.result.interface_return = load_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'transcribe':
                self.result.interface_return = transcribe(**kwargs)
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
