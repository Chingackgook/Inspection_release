# VideoLingo 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'VideoLingo/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/VideoLingo')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/VideoLingo')


# 可以在此位置后添加导包部分代码
import pandas as pd
import json
import concurrent.futures
from core.translate_lines import translate_lines
# add
from core.translate_lines import valid_translate_result
# end
from core.utils import *
from rich.console import Console
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from difflib import SequenceMatcher
from core.utils.models import *


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # 不需要初始化任何接口类的对象
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_file = False
        self.result.file_path = ''
        self.result.interface_return = None

    def run(self, name: str, **kwargs):
        try:
            if name == 'valid_translate_result':
                # 调用独立函数 valid_translate_result
                self.result.interface_return = valid_translate_result(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'translate_lines':
                # 调用独立函数 translate_lines
                self.result.interface_return = translate_lines(**kwargs)
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
adapter_additional_data['functions'].append('valid_translate_result')
adapter_additional_data['functions'].append('translate_lines')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
