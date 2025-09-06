# flair 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'flair/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/flair')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/flair')

# you can add your custom imports here
from flair.nn.model import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.classifier_obj = None  # Interface object for Classifier
        try:
            if interface_class_name == 'Classifier':
                # Create interface object for Classifier
                self.classifier_obj = Classifier.load(**kwargs)
                self.result.interface_return = self.classifier_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.classifier_obj = Classifier.load(**kwargs)
                self.result.interface_return = self.classifier_obj

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
            if dispatch_key == 'evaluate':
                # Call evaluate method
                self.result.interface_return = self.classifier_obj.evaluate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                # Call predict method
                self.result.interface_return = self.classifier_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_used_tokens':
                # Call get_used_tokens method
                self.result.interface_return = self.classifier_obj.get_used_tokens(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load':
                # Call load method (class method)
                self.result.interface_return = Classifier.load(**kwargs)
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
