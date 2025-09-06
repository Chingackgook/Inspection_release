# insightface 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'insightface/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/insightface/python-package')
# 以上是自动生成的代码，请勿修改


import numpy as np
from insightface.app import FaceAnalysis
from typing import Any, Dict
from insightface.data import get_image as ins_get_image
from insightface.utils import DEFAULT_MP_NAME, ensure_available

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.face_analysis = None

    def create_interface_objects(self, name = DEFAULT_MP_NAME, root: str = '~/.insightface', allowed_modules: list = None, **kwargs):
        try:
            self.face_analysis = FaceAnalysis(name=name, root=root, allowed_modules=allowed_modules, **kwargs)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
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
            if name == 'prepare':
                self.face_analysis.prepare(**kwargs)
            elif name == 'get':
                img = kwargs.pop('img', None)
                if img is not None:
                    faces = self.face_analysis.get(img, **kwargs)
                    self.result.interface_return = faces
                else:
                    raise ValueError("Image data must be provided for 'get' method.")
            elif name == 'draw_on':
                img = kwargs.pop('img', None)
                faces = kwargs.pop('faces', None)
                if img is not None and faces is not None:
                    rimg = self.face_analysis.draw_on(img, faces)
                    self.result.interface_return = rimg
                else:
                    raise ValueError("Image data and faces must be provided for 'draw_on' method.")
            else:
                raise ValueError(f"Method '{name}' is not recognized.")
            
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
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('prepare')
adapter_addtional_data['functions'].append('get')
adapter_addtional_data['functions'].append('draw_on')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
