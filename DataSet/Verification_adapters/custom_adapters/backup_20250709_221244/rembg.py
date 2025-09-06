# rembg 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'rembg/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/rembg')

# 可以在此位置后添加导包部分代码

from rembg.bg import remove
from rembg.session_factory import new_session 
from rembg.sessions import sessions_names
# DeadCodeFront end
from typing import Any, Dict, Union, Tuple
import numpy as np
from PIL import Image
import io

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        # 这里可以进行模型的加载操作
        # 由于文档未提供相关信息，直接pass
        pass

    def run(self, name: str, **kwargs):
        if name == 'remove':
            try:
                # 调用remove函数并传入kwargs
                output_data = remove(**kwargs)
                
                # 设置执行结果
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=output_data
                )
            except Exception as e:
                # 处理异常并设置失败原因
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason=str(e),
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
        else:
            # 如果name不匹配，设置失败结果
            self.result.set_result(
                fuc_name=name,
                is_success=False,
                fail_reason='Function not found',
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
adapter_additional_data['functions'].append('remove')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
