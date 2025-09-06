# spleeter 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'spleeter/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/spleeter')

# 可以在此位置后添加导包部分代码

import atexit
import os
from multiprocessing import Pool
from os.path import basename, dirname, join, splitext
from typing import Any, Dict, Generator, List, Optional
import numpy as np

from spleeter import SpleeterError
from spleeter.audio import Codec
from spleeter.audio.adapter import AudioAdapter
from spleeter.audio.convertor import to_stereo
from spleeter.model import EstimatorSpecBuilder, InputProviderFactory, model_fn
from spleeter.model.provider import ModelProvider
from spleeter.types import AudioDescriptor
from spleeter.utils.configuration import load_configuration

from spleeter.separator import Separator
# DeadCodeFront end

from typing import Any, Dict
from abc import ABC
import tensorflow as tf

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.separator = None

    def create_interface_objects(self, **kwargs):
        try:
            self.separator = Separator(**kwargs)
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
            if name == 'separate':
                output = self.separator.separate(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == 'separate_to_file':
                self.separator.separate_to_file(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=True,
                    file_path=kwargs.get('destination', ''),
                    except_data=None,
                    interface_return=None
                )
            elif name == 'save_to_file':
                self.separator.save_to_file(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=True,
                    file_path=kwargs.get('destination', ''),
                    except_data=None,
                    interface_return=None
                )
            elif name == 'join':
                self.separator.join(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == '_get_prediction_generator':
                output = self.separator._get_prediction_generator(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == '_get_input_provider':
                output = self.separator._get_input_provider()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == '_get_features':
                output = self.separator._get_features()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == '_get_builder':
                output = self.separator._get_builder()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == '_get_session':
                output = self.separator._get_session()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == '_separate_tensorflow':
                output = self.separator._separate_tensorflow(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            elif name == 'create_estimator':
                output = create_estimator(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=output,
                    interface_return=output
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='Method not found',
                    is_file=False,
                    file_path='',
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
adapter_additional_data['functions'].append('separate')
adapter_additional_data['functions'].append('separate_to_file')
adapter_additional_data['functions'].append('save_to_file')
adapter_additional_data['functions'].append('join')
adapter_additional_data['functions'].append('_get_prediction_generator')
adapter_additional_data['functions'].append('_get_input_provider')
adapter_additional_data['functions'].append('_get_features')
adapter_additional_data['functions'].append('_get_builder')
adapter_additional_data['functions'].append('_get_session')
adapter_additional_data['functions'].append('_separate_tensorflow')
adapter_additional_data['functions'].append('create_estimator')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
