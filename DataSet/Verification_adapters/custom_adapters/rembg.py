# rembg 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'rembg/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/rembg')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/rembg')

# you can add your custom imports here
from rembg.bg import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        """
        Method to initialize interface objects. This should be implemented in CustomAdapter,
        which creates interface objects based on `interface_class_name` and `kwargs`.
        The result should be stored in `self.result`.
        """
        try:
            if interface_class_name == 'remove':
                # No specific class object to create for the top-level function
                self.result.interface_return = None  # No object to return for top-level function
            else:
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.result.interface_return = None  # No object to return for top-level function

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
        Entry point for execution. Executes corresponding methods/functions based on `dispatch_key`,
        and stores the result in `self.result`.
        """
        try:
            if dispatch_key == 'remove':
                # Call the top-level function remove
                self.result.interface_return = remove(**kwargs)
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
