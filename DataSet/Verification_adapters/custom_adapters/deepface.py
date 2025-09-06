# deepface 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'deepface/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/deepface')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/deepface')

# you can add your custom imports here
from deepface.DeepFace import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # No interface class objects need to be initialized as all are top-level functions
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_file = False
        self.result.file_path = ''
        self.result.interface_return = None

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'build_model':
                self.result.interface_return = build_model(**kwargs)
            elif dispatch_key == 'verify':
                self.result.interface_return = verify(**kwargs)
            elif dispatch_key == 'analyze':
                self.result.interface_return = analyze(**kwargs)
            elif dispatch_key == 'find':
                self.result.interface_return = find(**kwargs)
            elif dispatch_key == 'represent':
                self.result.interface_return = represent(**kwargs)
            elif dispatch_key == 'stream':
                self.result.interface_return = stream(**kwargs)
            elif dispatch_key == 'extract_faces':
                self.result.interface_return = extract_faces(**kwargs)
            elif dispatch_key == 'cli':
                self.result.interface_return = cli(**kwargs)
            elif dispatch_key == 'detectFace':
                self.result.interface_return = detectFace(**kwargs)
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = dispatch_key

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
