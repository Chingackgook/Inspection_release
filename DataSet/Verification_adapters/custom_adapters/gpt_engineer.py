# gpt_engineer 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gpt_engineer/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/gpt-engineer')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/gpt-engineer')

# you can add your custom imports here
from gpt_engineer.core.ai import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Placeholder for the AI class object
        try:
            if interface_class_name == 'AI':
                # Create interface object for AI
                self.class1_obj = AI(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = AI(**kwargs)
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
            if dispatch_key == 'start':
                # Call start method from AI
                self.result.interface_return = self.class1_obj.start(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'next':
                # Call next method from AI
                self.result.interface_return = self.class1_obj.next(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'backoff_inference':
                # Call backoff_inference method from AI
                self.result.interface_return = self.class1_obj.backoff_inference(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'serialize_messages':
                # Call serialize_messages method from AI
                self.result.interface_return = self.class1_obj.serialize_messages(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'deserialize_messages':
                # Call deserialize_messages method from AI
                self.result.interface_return = self.class1_obj.deserialize_messages(**kwargs)
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
