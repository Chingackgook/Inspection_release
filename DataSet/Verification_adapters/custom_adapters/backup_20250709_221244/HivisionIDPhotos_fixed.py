# HivisionIDPhotos_fixed 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'HivisionIDPhotos_fixed/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/HivisionIDPhotos')
# 以上是自动生成的代码，请勿修改



from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np
from hivision import IDCreator

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.id_creator = None

    def create_interface_objects(self, **kwargs):
        """
        初始化IDCreator类的实例
        """
        self.id_creator = IDCreator()
        # 这里可以根据kwargs设置其他参数，例如回调函数等
        # 例如: self.id_creator.before_all = kwargs.get('before_all', None)
        self.result.set_result(
            fuc_name='create_interface_objects',
            is_success=True,
            fail_reason='',
            is_file=False,
            file_path='',
            except_data=None,
            interface_return=self.id_creator
        )

    def run(self, name: str, **kwargs):
        """
        执行IDCreator的相关方法
        """
        if name == '__call__':
            # 调用IDCreator的__call__方法
            try:
                size = kwargs.get('size', None)
                face_alignment = kwargs.get('face_alignment', False)
                image = kwargs.get('image', None)
                result = self.id_creator(image, size=size, face_alignment=face_alignment)
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
        elif name == 'before_all':
            # 处理before_all回调
            if hasattr(self.id_creator, 'before_all'):
                self.id_creator.before_all(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            else:
                raise ValueError("before_all handler not set.")
        elif name == 'after_matting':
            # 处理after_matting回调
            if hasattr(self.id_creator, 'after_matting'):
                self.id_creator.after_matting(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            else:
                raise ValueError("after_matting handler not set.")
        elif name == 'after_detect':
            # 处理after_detect回调
            if hasattr(self.id_creator, 'after_detect'):
                self.id_creator.after_detect(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            else:
                raise ValueError("after_detect handler not set.")
        elif name == 'after_all':
            # 处理after_all回调
            if hasattr(self.id_creator, 'after_all'):
                self.id_creator.after_all(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            else:
                raise ValueError("after_all handler not set.")
        else:
            raise ValueError(f"Unsupported method name: {name}")

# 外层调用示例，供参考
# adapter = CustomAdapter()
# adapter.create_interface_objects()
# result = adapter.run(name='__call__', image=input_image, size=(413, 295), face_alignment=True)
# print(adapter.result)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('__call__')
adapter_addtional_data['functions'].append('before_all')
adapter_addtional_data['functions'].append('after_matting')
adapter_addtional_data['functions'].append('after_detect')
adapter_addtional_data['functions'].append('after_all')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
