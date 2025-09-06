# yolov10 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'yolov10/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/yolov10')

# 可以在此位置后添加导包部分代码
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils import TQDM
from ultralytics.utils.files import increment_path
from ultralytics.data import YOLODataset
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import LOGGER
from ultralytics import SAM
from tqdm import tqdm


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.sam_obj = None  # SAM 类的接口对象

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'SAM':
                # 创建 SAM 接口对象
                self.sam_obj = SAM(**kwargs)
                self.result.interface_return = self.sam_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象
                self.sam_obj = SAM(**kwargs)
                self.result.interface_return = self.sam_obj
            
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
            if name == 'predict':
                # 执行 SAM 的 predict 方法
                self.result.interface_return = self.sam_obj.predict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == '__call__':
                # 执行 SAM 的 __call__ 方法
                self.result.interface_return = self.sam_obj.__call__(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'info':
                # 执行 SAM 的 info 方法
                self.result.interface_return = self.sam_obj.info(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            else:
                raise ValueError(f"Unknown method name: {name}")

        except Exception as e:
            self.result.fuc_name = name
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行方法 {name} 失败: {e}")

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('SAM')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('predict')
adapter_additional_data['functions'].append('__call__')
adapter_additional_data['functions'].append('info')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
