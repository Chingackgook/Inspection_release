# audiocraft 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'audiocraft/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/audiocraft')
# 以上是自动生成的代码，请勿修改


import torch
from audiocraft.models import MusicGen

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_interface_objects(self, name='facebook/musicgen-melody', device=None, **kwargs):
        try:
            self.model = MusicGen.get_pretrained(name=name, device=device)
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
            if name == 'set_generation_params':
                self.model.set_generation_params(**kwargs)
            elif name == 'set_style_conditioner_params':
                self.model.set_style_conditioner_params(**kwargs)
            elif name == 'generate':
                generated_audio = self.model.generate(**kwargs)
                self.result.set_result(
                    fuc_name='generate',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=generated_audio,
                    interface_return=generated_audio
                )
                return
            elif name == 'generate_with_chroma':
                generated_audio = self.model.generate_with_chroma(**kwargs)
                self.result.set_result(
                    fuc_name='generate_with_chroma',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=generated_audio,
                    interface_return=generated_audio
                )
                return
            elif name == '_prepare_tokens_and_attributes':
                attributes, tokens = self.model._prepare_tokens_and_attributes(**kwargs)
                self.result.set_result(
                    fuc_name='_prepare_tokens_and_attributes',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=(attributes, tokens),
                    interface_return=(attributes, tokens)
                )
                return
            elif name == '_generate_tokens':
                tokens = self.model._generate_tokens(**kwargs)
                self.result.set_result(
                    fuc_name='_generate_tokens',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=tokens,
                    interface_return=tokens
                )
                return
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
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
adapter_additional_data['functions'].append('set_generation_params')
adapter_additional_data['functions'].append('set_style_conditioner_params')
adapter_additional_data['functions'].append('generate_with_chroma')
adapter_additional_data['functions'].append('_prepare_tokens_and_attributes')
adapter_additional_data['functions'].append('_generate_tokens')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
