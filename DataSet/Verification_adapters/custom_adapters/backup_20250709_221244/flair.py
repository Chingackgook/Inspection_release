# flair 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'flair/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/flair')
# 以上是自动生成的代码，请勿修改


from flair.data import Sentence
from flair.nn import Classifier
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, Tuple
from pathlib import Path

class ExecutionResult:
    def __init__(self):
        self.fuc_name = ''
        self.is_success = False
        self.fail_reason = ''
        self.is_file = False
        self.file_path = ''
        self.except_data = None
        self.interface_return = None

    def set_result(
        self, 
        fuc_name: str,
        is_success: bool,
        fail_reason: str,
        is_file: bool, 
        file_path: str, 
        except_data: Any, 
        interface_return: Any,
    ):
        self.fuc_name = fuc_name
        self.is_success = is_success
        self.fail_reason = fail_reason
        self.is_file = is_file
        self.file_path = file_path
        self.interface_return = interface_return
        self.except_data = except_data

class BaseAdapter(ABC):
    def __init__(self):
        self.result: ExecutionResult = ExecutionResult()
    
    @abstractmethod
    def create_interface_objects(self, **kwargs):
        pass

    @abstractmethod
    def run(self, name: str, **kwargs):
        pass

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.classifier = None

    def create_interface_objects(self, model_path: Union[str, Path]):
        try:
            self.classifier = Classifier.load(model_path)
            self.result.set_result('create_interface_objects', True, '', False, '', None, self.classifier)
        except Exception as e:
            self.result.set_result('create_interface_objects', False, str(e), False, '', None, None)

    def run(self, name: str, **kwargs):
        try:
            if name == 'evaluate':
                result = self.classifier.evaluate(**kwargs)
            elif name == 'predict':
                result = self.classifier.predict(**kwargs)
            elif name == '_print_predictions':
                result = self.classifier._print_predictions(**kwargs)
            elif name == 'get_used_tokens':
                result = self.classifier.get_used_tokens(**kwargs)
            elif name == 'load':
                result = Classifier.load(**kwargs)
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(name, True, '', False, '', None, result)
        except Exception as e:
            self.result.set_result(name, False, str(e), False, '', None, None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('evaluate')
adapter_additional_data['functions'].append('predict')
adapter_additional_data['functions'].append('_print_predictions')
adapter_additional_data['functions'].append('get_used_tokens')
adapter_additional_data['functions'].append('load')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
