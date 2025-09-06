# Real_Time_Voice_Cloning 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Real_Time_Voice_Cloning/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Real-Time-Voice-Cloning')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Real-Time-Voice-Cloning')

# you can add your custom imports here
from synthesizer.inference import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'Synthesizer':
                # Create interface object
                self.synthesizer_obj = Synthesizer(**kwargs)
                self.result.interface_return = self.synthesizer_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.synthesizer_obj = Synthesizer(**kwargs)
                self.result.interface_return = self.synthesizer_obj

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
            if dispatch_key == 'is_loaded':
                # Call the is_loaded method
                self.result.interface_return = self.synthesizer_obj.is_loaded()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load':
                # Call the load method
                self.synthesizer_obj.load()
                self.result.interface_return = None  # load does not return a value
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'synthesize_spectrograms':
                # Call the synthesize_spectrograms method
                self.result.interface_return = self.synthesizer_obj.synthesize_spectrograms(**kwargs)
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
