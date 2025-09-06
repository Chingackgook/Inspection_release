# point_e 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'point_e/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/point-e')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/point-e')

# you can add your custom imports here
from point_e.diffusion.sampler import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'PointCloudSampler':
                # Create interface object for PointCloudSampler
                self.point_cloud_sampler = PointCloudSampler(**kwargs)
                self.result.interface_return = self.point_cloud_sampler
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.point_cloud_sampler = PointCloudSampler(**kwargs)
                self.result.interface_return = self.point_cloud_sampler

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
            if dispatch_key == 'num_stages':
                # Call num_stages method
                self.result.interface_return = self.point_cloud_sampler.num_stages(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'sample_batch':
                # Call sample_batch method
                self.result.interface_return = self.point_cloud_sampler.sample_batch(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'sample_batch_progressive':
                # Call sample_batch_progressive method
                self.result.interface_return = self.point_cloud_sampler.sample_batch_progressive(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'combine':
                # Call combine method
                self.result.interface_return = self.point_cloud_sampler.combine(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'split_model_output':
                # Call split_model_output method
                self.result.interface_return = self.point_cloud_sampler.split_model_output(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'output_to_point_clouds':
                # Call output_to_point_clouds method
                self.result.interface_return = self.point_cloud_sampler.output_to_point_clouds(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'with_options':
                # Call with_options method
                self.result.interface_return = self.point_cloud_sampler.with_options(**kwargs)
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
