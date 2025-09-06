# deepface 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'deepface/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/deepface')
# 以上是自动生成的代码，请勿修改


from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd

# 假设这些函数和类已经在其他地方实现
from deepface.DeepFace import build_model, verify, analyze, find, represent, stream, extract_faces, Logger

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_interface_objects(self, model_name: str, task: str = "facial_recognition"):
        try:
            self.model = build_model(model_name, task)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.model
            )
        except Exception as e:
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

    def run(self, name: str, **kwargs):
        try:
            if name == 'verify':
                self.result.interface_return = verify(**kwargs)
            elif name == 'analyze':
                self.result.interface_return = analyze(**kwargs)
            elif name == 'find':
                self.result.interface_return = find(**kwargs)
            elif name == 'represent':
                self.result.interface_return = represent(**kwargs)
            elif name == 'stream':
                stream(**kwargs)  # stream doesn't return a value
                self.result.interface_return = None
            elif name == 'extract_faces':
                self.result.interface_return = extract_faces(**kwargs)
            elif name == 'cli':
                #cli()  # cli doesn't return a value
                self.result.interface_return = None
            else:
                raise ValueError(f"Function {name} is not recognized.")
            
            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.result.interface_return
            )
        except Exception as e:
            self.result.set_result(
                fuc_name=name,
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('verify')
adapter_additional_data['functions'].append('analyze')
adapter_additional_data['functions'].append('find')
adapter_additional_data['functions'].append('represent')
adapter_additional_data['functions'].append('stream')
adapter_additional_data['functions'].append('extract_faces')
adapter_additional_data['functions'].append('cli')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
