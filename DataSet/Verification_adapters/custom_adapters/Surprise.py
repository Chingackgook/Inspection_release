# Surprise 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Surprise/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Surprise')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Surprise')

# you can add your custom imports here
from surprise.prediction_algorithms.knns import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'KNNBasic':
                # Create interface object
                self.knn_basic_obj = KNNBasic(**kwargs)
                self.result.interface_return = self.knn_basic_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.knn_basic_obj = KNNBasic(**kwargs)
                self.result.interface_return = self.knn_basic_obj

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
                self.result.interface_return = self.knn_basic_obj.fit(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'predict':
                self.result.interface_return = self.knn_basic_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'test':
                self.result.interface_return = self.knn_basic_obj.test(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'compute_baselines':
                self.result.interface_return = self.knn_basic_obj.compute_baselines()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'compute_similarities':
                self.result.interface_return = self.knn_basic_obj.compute_similarities()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_neighbors':
                self.result.interface_return = self.knn_basic_obj.get_neighbors(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'switch':
                self.result.interface_return = self.knn_basic_obj.switch(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'estimate':
                self.result.interface_return = self.knn_basic_obj.estimate(**kwargs)
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
