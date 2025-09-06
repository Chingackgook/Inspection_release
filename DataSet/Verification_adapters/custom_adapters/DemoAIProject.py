# DemoAIProject 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'DemoAIProject/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/Inspection/Demo/DemoAIProject/')
os.chdir('/mnt/autor_name/Inspection/Demo/DemoAIProject/')

# you can add your custom imports here
from ai_agent import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'AIAgent':
                # Create interface object
                self.agent = AIAgent(**kwargs)
                self.result.interface_return = self.agent
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.agent = AIAgent(**kwargs)
                self.result.interface_return = self.agent

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'chat':
                self.result.interface_return = self.agent.chat(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'generate_text':
                self.result.interface_return = self.agent.generate_text(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'analyze_text':
                self.result.interface_return = self.agent.analyze_text(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'set_model':
                self.result.interface_return = self.agent.set_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_model_info':
                self.result.interface_return = self.agent.get_model_info()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
