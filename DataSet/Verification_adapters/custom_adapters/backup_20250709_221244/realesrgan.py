# realesrgan 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'realesrgan/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/Real-ESRGAN')
# 以上是自动生成的代码，请勿修改


import numpy as np
import torch
from realesrgan import RealESRGANer
from abc import ABC, abstractmethod
from typing import Any

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.realesrgan = None

    def create_interface_objects(self, model_path: str, scale: int = 4, **kwargs):
        try:
            self.realesrgan = RealESRGANer(scale=scale, model_path=model_path, **kwargs)
            self.result.set_result('create_interface_objects', True, '', False, '', None, None)
        except Exception as e:
            self.result.set_result('create_interface_objects', False, str(e), False, '', None, None)

    def run(self, name: str, **kwargs):
        try:
            if name == 'pre_process':
                self.realesrgan.pre_process(**kwargs)
                self.result.set_result(name, True, '', False, '', None, None)
            elif name == 'process':
                self.realesrgan.process(**kwargs)
                self.result.set_result(name, True, '', False, '', None, None)
            elif name == 'tile_process':
                self.realesrgan.tile_process(**kwargs)
                self.result.set_result(name, True, '', False, '', None, None)
            elif name == 'post_process':
                output = self.realesrgan.post_process(**kwargs)
                self.result.set_result(name, True, '', False, '', output, output)
            elif name == 'enhance':
                output, img_mode = self.realesrgan.enhance(**kwargs)
                self.result.set_result(name, True, '', False, '', output, (output, img_mode))
            elif name == 'dni':
                output = self.realesrgan.dni(**kwargs)
                self.result.set_result(name, True, '', False, '', output, output)
            else:
                self.result.set_result(name, False, 'Method not found', False, '', None, None)
        except Exception as e:
            self.result.set_result(name, False, str(e), False, '', None, None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('pre_process')
adapter_additional_data['functions'].append('process')
adapter_additional_data['functions'].append('tile_process')
adapter_additional_data['functions'].append('post_process')
adapter_additional_data['functions'].append('enhance')
adapter_additional_data['functions'].append('dni')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
