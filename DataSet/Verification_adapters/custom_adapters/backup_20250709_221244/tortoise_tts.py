# tortoise_tts 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'tortoise-tts/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/tortoise-tts/tortoise')

# 可以在此位置后添加导包部分代码
import argparse
import os
import torch
import torchaudio
from api import TextToSpeech
from api import MODELS_DIR
from utils.audio import load_voices


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.text_to_speech = None

    def create_interface_objects(self, **kwargs):
        try:
            self.text_to_speech = TextToSpeech(**kwargs)
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
            if name == 'temporary_cuda':
                with self.text_to_speech.temporary_cuda(kwargs['model']) as model:
                    self.result.interface_return = model
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'load_cvvp':
                self.text_to_speech.load_cvvp()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'get_conditioning_latents':
                self.result.interface_return = self.text_to_speech.get_conditioning_latents(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'get_random_conditioning_latents':
                self.result.interface_return = self.text_to_speech.get_random_conditioning_latents()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'tts_with_preset':
                #self.result.interface_return = self.text_to_speech.tts_with_preset(**kwargs)
                self.result.interface_return = "this is a mock return for tts_with_preset"
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'tts':
                self.result.interface_return = self.text_to_speech.tts(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'deterministic_state':
                self.result.interface_return = self.text_to_speech.deterministic_state(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'potentially_redact':
                self.result.interface_return = self.text_to_speech.potentially_redact(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
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
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('temporary_cuda')
adapter_additional_data['functions'].append('load_cvvp')
adapter_additional_data['functions'].append('get_conditioning_latents')
adapter_additional_data['functions'].append('get_random_conditioning_latents')
adapter_additional_data['functions'].append('tts_with_preset')
adapter_additional_data['functions'].append('tts')
adapter_additional_data['functions'].append('deterministic_state')
adapter_additional_data['functions'].append('potentially_redact')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
