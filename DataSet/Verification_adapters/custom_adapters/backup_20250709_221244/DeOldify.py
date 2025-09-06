# DeOldify 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'DeOldify/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/DeOldify')

# 可以在此位置后添加导包部分代码
import os
import statistics
from fastai import *
from deoldify.visualize import *
from deoldify.visualize import ModelImageVisualizer
import cv2
from fid.fid_score import *
from fid.inception import *
import imageio
import warnings


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.visualizer = None  # ModelImageVisualizer 的接口对象

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'ModelImageVisualizer':
                # 创建接口对象
                self.visualizer = ModelImageVisualizer(**kwargs)
                self.result.interface_return = self.visualizer
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象
                self.visualizer = ModelImageVisualizer(**kwargs)
                self.result.interface_return = self.visualizer
            
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 创建接口对象失败: {e}")

    def run(self, name: str, **kwargs):
        try:
            if name == 'get_watermarked':
                # 执行独立函数
                self.result.interface_return = get_watermarked(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'plot_transformed_image_from_url':
                # 执行 ModelImageVisualizer 的方法
                self.result.interface_return = self.visualizer.plot_transformed_image_from_url(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'plot_transformed_image':
                # 执行 ModelImageVisualizer 的方法
                self.result.interface_return = self.visualizer.plot_transformed_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'get_transformed_image':
                # 执行 ModelImageVisualizer 的方法
                self.result.interface_return = self.visualizer.get_transformed_image(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            else:
                raise ValueError(f"Unknown method name: {name}")

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行方法 {name} 失败: {e}")

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('ModelImageVisualizer')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('get_watermarked')
adapter_additional_data['functions'].append('plot_transformed_image_from_url')
adapter_additional_data['functions'].append('plot_transformed_image')
adapter_additional_data['functions'].append('get_transformed_image')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
