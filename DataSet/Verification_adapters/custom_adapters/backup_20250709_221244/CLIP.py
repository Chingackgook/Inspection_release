# CLIP 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'CLIP/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('')

# 可以在此位置后添加导包部分代码
import numpy as np
import pytest
import torch
from PIL import Image
from clip import available_models, load, tokenize


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        """
        初始化接口对象的方法，根据interface_class_name和kwargs创建接口对象，由子类CustomAdapter实现，
        将结果存储在self.result中，如不存在接口类，pass即可。
        """
        self.tokenizer = None
        try:
            if interface_class_name == '_Tokenizer':
                # 创建接口对象
                self.tokenizer = _Tokenizer()
                self.result.interface_return = self.tokenizer
            else:
                pass
            
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
        """
        执行方法的入口，由子类CustomAdapter实现，根据name执行对应的方法，并将结果存储在self.result中
        请使用if name == 'xxx' 来判断执行什么方法
        """
        try:
            if name == 'available_models':
                # 执行函数接口
                self.result.interface_return = available_models()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'load':
                # 执行加载模型的函数
                model_name = kwargs.get('name', '')
                device = kwargs.get('device', None)
                jit = kwargs.get('jit', False)
                download_root = kwargs.get('download_root', None)
                self.result.interface_return = load(model_name, device=device, jit=jit, download_root=download_root)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'tokenize':
                # 执行标记化的函数
                texts = kwargs.get('texts', [])
                context_length = kwargs.get('context_length', 77)
                truncate = kwargs.get('truncate', False)
                self.result.interface_return = tokenize(texts, context_length=context_length, truncate=truncate)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == '_Tokenizer_encode':
                # 执行Tokenizer的encode方法
                if self.tokenizer is not None:
                    texts = kwargs.get('texts', [])
                    self.result.interface_return = self.tokenizer.encode(texts)
                    self.result.is_success = True
                    self.result.fail_reason = ''
                    self.result.fuc_name = name
                else:
                    raise ValueError("Tokenizer not initialized.")
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
adapter_additional_data['functions'].append('_Tokenizer')
adapter_additional_data['functions'].append('xxx')
adapter_additional_data['functions'].append('available_models')
adapter_additional_data['functions'].append('load')
adapter_additional_data['functions'].append('tokenize')
adapter_additional_data['functions'].append('_Tokenizer_encode')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
