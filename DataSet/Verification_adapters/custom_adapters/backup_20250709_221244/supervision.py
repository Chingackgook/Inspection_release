# supervision 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'supervision/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/supervision')

# 可以在此位置后添加导包部分代码
import argparse
from collections import defaultdict
from collections import deque
import cv2
import numpy as np
from super_gradients.common.object_names import Models
from super_gradients.training import models
from supervision.detection.core import Detections
import supervision as sv


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.detections_instance = None  # Detections 类的实例

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'Detections':
                # 创建 Detections 接口对象
                self.detections_instance = Detections(**kwargs)
                self.result.interface_return = self.detections_instance
            elif interface_class_name == '':
                # 如果缺省，创建默认 Detections 接口对象
                self.detections_instance = Detections(**kwargs)
                self.result.interface_return = self.detections_instance
            
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
            if name == '__len__':
                self.result.interface_return = len(self.detections_instance)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == '__iter__':
                self.result.interface_return = list(iter(self.detections_instance))
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == '__eq__':
                other = kwargs.get('other')
                self.result.interface_return = self.detections_instance == other
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'is_empty':
                self.result.interface_return = self.detections_instance.is_empty()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'area':
                self.result.interface_return = self.detections_instance.area
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'box_area':
                self.result.interface_return = self.detections_instance.box_area
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'merge':
                detections_list = kwargs.get('detections_list')
                self.result.interface_return = Detections.merge(detections_list)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'with_nms':
                threshold = kwargs.get('threshold', 0.5)
                class_agnostic = kwargs.get('class_agnostic', False)
                self.result.interface_return = self.detections_instance.with_nms(threshold, class_agnostic)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'with_nmm':
                threshold = kwargs.get('threshold', 0.5)
                class_agnostic = kwargs.get('class_agnostic', False)
                self.result.interface_return = self.detections_instance.with_nmm(threshold, class_agnostic)
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
adapter_additional_data['functions'].append('Detections')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('__len__')
adapter_additional_data['functions'].append('__iter__')
adapter_additional_data['functions'].append('__eq__')
adapter_additional_data['functions'].append('is_empty')
adapter_additional_data['functions'].append('area')
adapter_additional_data['functions'].append('box_area')
adapter_additional_data['functions'].append('merge')
adapter_additional_data['functions'].append('with_nms')
adapter_additional_data['functions'].append('with_nmm')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
