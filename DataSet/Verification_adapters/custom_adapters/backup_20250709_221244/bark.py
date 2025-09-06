# bark 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'bark/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/bark')
# 以上是自动生成的代码，请勿修改



from abc import ABC
from typing import Any, Dict
import numpy as np
from bark.api import generate_audio
from bark.generation import SAMPLE_RATE
from bark.api import text_to_semantic
from bark.api import save_as_prompt
from bark.api import semantic_to_waveform

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        pass

    def run(self, name: str, **kwargs):
        try:
            if name == 'generate_audio':
                audio_array = generate_audio(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=audio_array,
                    interface_return=audio_array
                )
            elif name == 'text_to_semantic':
                semantic_array = text_to_semantic(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=semantic_array,
                    interface_return=semantic_array
                )
            elif name == 'semantic_to_waveform':
                audio_array = semantic_to_waveform(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=audio_array,
                    interface_return=audio_array
                )
            elif name == 'save_as_prompt':
                save_as_prompt(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=True,
                    file_path=kwargs.get('filepath', ''),
                    except_data=None,
                    interface_return=None
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='Function not found',
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
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('generate_audio')
adapter_addtional_data['functions'].append('text_to_semantic')
adapter_addtional_data['functions'].append('semantic_to_waveform')
adapter_addtional_data['functions'].append('save_as_prompt')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
