# DragGAN 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'DragGAN/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/DragGAN')

# 可以在此位置后添加导包部分代码
from gen_images import generate_images, parse_range, parse_vec2, make_transform
# DeadCodeFront end

import numpy as np
from typing import Any, Dict, List, Tuple, Union

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        pass

    def run(self, name: str, **kwargs):
        if name == 'parse_range':
            try:
                result = parse_range(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=result,
                    interface_return=result
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

        elif name == 'parse_vec2':
            try:
                result = parse_vec2(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=result,
                    interface_return=result
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

        elif name == 'make_transform':
            try:
                result = make_transform(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=result,
                    interface_return=result
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

        elif name == 'generate_images':
            try:
                generate_images.callback(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=True,
                    file_path=kwargs.get('outdir', ''),
                    except_data=None,
                    interface_return=None
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
adapter_additional_data['functions'].append('parse_range')
adapter_additional_data['functions'].append('parse_vec2')
adapter_additional_data['functions'].append('make_transform')
adapter_additional_data['functions'].append('generate_images')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
