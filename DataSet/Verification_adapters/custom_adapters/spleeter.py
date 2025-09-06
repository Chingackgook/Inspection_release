# spleeter 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'spleeter/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/spleeter')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/spleeter')

# you can add your custom imports here
from spleeter.separator import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        """
        Method to initialize interface objects. This creates an instance of the Separator class.
        """
        self.separator_obj = None  # Interface object for Separator
        try:
            if interface_class_name == 'Separator':
                # Create interface object
                self.separator_obj = Separator(**kwargs)
                self.result.interface_return = self.separator_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.separator_obj = Separator(**kwargs)
                self.result.interface_return = self.separator_obj

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
        """
        Entry point for execution. Executes corresponding methods/functions based on `dispatch_key`.
        """
        try:
            if dispatch_key == 'separate':
                # Call separate method from Separator
                self.result.interface_return = self.separator_obj.separate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'separate_to_file':
                # Call separate_to_file method from Separator
                self.separator_obj.separate_to_file(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'save_to_file':
                # Call save_to_file method from Separator
                self.separator_obj.save_to_file(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'join':
                # Call join method from Separator
                self.separator_obj.join(**kwargs)
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
