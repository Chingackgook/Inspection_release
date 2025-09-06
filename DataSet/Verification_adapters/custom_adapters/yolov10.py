# yolov10 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'yolov10/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/yolov10')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/yolov10')

# you can add your custom imports here
from ultralytics.models.sam.model import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Placeholder for SAM object
        self.class2_obj = None  # Placeholder for Model object

        try:
            if interface_class_name == 'SAM':
                # Create SAM interface object
                self.class1_obj = SAM(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'Model':
                self.class2_obj = Model(**kwargs)
                self.result.interface_return = self.class2_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object (only use if there is a single interface class)
                self.class1_obj = SAM(**kwargs)
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
            if dispatch_key == 'is_triton_model':
                self.result.interface_return = is_triton_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'is_hub_model':
                self.result.interface_return = is_hub_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                self.result.interface_return = self.class1_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'info':
                self.result.interface_return = self.class1_obj.info(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'task_map':
                self.result.interface_return = self.class1_obj.task_map()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'load':
                self.result.interface_return = self.class2_obj.load(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'save':
                self.result.interface_return = self.class2_obj.save(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                self.result.interface_return = self.class2_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'track':
                self.result.interface_return = self.class2_obj.track(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'val':
                self.result.interface_return = self.class2_obj.val(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'benchmark':
                self.result.interface_return = self.class2_obj.benchmark(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'export':
                self.result.interface_return = self.class2_obj.export(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'train':
                self.result.interface_return = self.class2_obj.train(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'tune':
                self.result.interface_return = self.class2_obj.tune(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'names':
                self.result.interface_return = self.class2_obj.names()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'device':
                self.result.interface_return = self.class2_obj.device()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'transforms':
                self.result.interface_return = self.class2_obj.transforms()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'add_callback':
                self.result.interface_return = self.class2_obj.add_callback(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'clear_callback':
                self.result.interface_return = self.class2_obj.clear_callback(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'reset_callbacks':
                self.result.interface_return = self.class2_obj.reset_callbacks()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'task_map':
                self.result.interface_return = self.class2_obj.task_map()
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
