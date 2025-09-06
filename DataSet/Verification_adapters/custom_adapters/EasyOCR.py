# EasyOCR 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'EasyOCR/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/EasyOCR')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/EasyOCR')

# you can add your custom imports here
from easyocr.easyocr import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.reader_obj = None  # Interface object for Reader
        try:
            if interface_class_name == 'Reader':
                # Create interface object
                self.reader_obj = Reader(**kwargs)
                self.result.interface_return = self.reader_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.reader_obj = Reader(**kwargs)
                self.result.interface_return = self.reader_obj

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
            if dispatch_key == 'detect':
                self.result.interface_return = self.reader_obj.detect(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'recognize':
                self.result.interface_return = self.reader_obj.recognize(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'readtext':
                self.result.interface_return = self.reader_obj.readtext(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'readtextlang':
                self.result.interface_return = self.reader_obj.readtextlang(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'readtext_batched':
                self.result.interface_return = self.reader_obj.readtext_batched(**kwargs)
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
