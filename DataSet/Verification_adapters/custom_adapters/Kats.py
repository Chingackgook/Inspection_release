# Kats 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Kats/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Kats')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Kats')

# you can add your custom imports here
from kats.models.sarima import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'SARIMAModel':
                # Create interface object for SARIMAModel
                self.model_obj = SARIMAModel(**kwargs)
                self.result.interface_return = self.model_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.model_obj = SARIMAModel(**kwargs)
                self.result.interface_return = self.model_obj

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
            if dispatch_key == 'fit':
                # Call fit method from SARIMAModel
                self.result.interface_return = self.model_obj.fit(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                # Call predict method from SARIMAModel
                self.result.interface_return = self.model_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_parameter_search_space':
                # Call get_parameter_search_space method from SARIMAModel
                self.result.interface_return = self.model_obj.get_parameter_search_space()
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
