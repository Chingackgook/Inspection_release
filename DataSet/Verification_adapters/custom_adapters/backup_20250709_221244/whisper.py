# whisper 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'whisper/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/whisper')
# 以上是自动生成的代码，请勿修改



import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from whisper import available_models, load_model, transcribe ,log_mel_spectrogram,load_audio, DecodingOptions

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_interface_objects(self, name: str, device: Optional[Union[str, torch.device]] = None, 
                   download_root: str = None, in_memory: bool = False):
        try:
            self.model = load_model(name, device=device, download_root=download_root, in_memory=in_memory)
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
            if name == 'detect_language':
                mel_segment = kwargs.get('mel_segment')
                result = self.model.detect_language(mel_segment)
            elif name == 'decode':
                segment = kwargs.get('segment')
                options = kwargs.get('options')
                result = self.model.decode(segment, options)
            elif name == 'load_state_dict':
                state_dict = kwargs.get('state_dict')
                self.model.load_state_dict(state_dict)
                result = None
            elif name == 'set_alignment_heads':
                alignment_heads = kwargs.get('alignment_heads')
                self.model.set_alignment_heads(alignment_heads)
                result = None
            elif name == 'transcribe':
                # 如果有model参数，使用传入的model
                if 'model' in kwargs:
                    model = kwargs.pop('model')
                else:
                    model = self.model
                result = transcribe(model, **kwargs)
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
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

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('detect_language')
adapter_addtional_data['functions'].append('decode')
adapter_addtional_data['functions'].append('load_state_dict')
adapter_addtional_data['functions'].append('set_alignment_heads')
adapter_addtional_data['functions'].append('transcribe')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
