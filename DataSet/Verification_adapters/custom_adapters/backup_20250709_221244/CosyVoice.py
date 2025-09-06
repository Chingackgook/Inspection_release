# CosyVoice 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'CosyVoice/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/CosyVoice')
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/CosyVoice/third_party/Matcha_TTS')
# 可以在此位置后添加导包部分代码




# DeadCodeFront end

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from abc import ABC, abstractmethod
from typing import Any, Dict

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.cosyvoice = None

    def create_interface_objects(self, model_dir: str, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False):
        try:
            self.cosyvoice = CosyVoice2(model_dir, load_jit, load_trt, fp16, use_flow_cache)
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
            if name == 'list_available_spks':
                result = self.cosyvoice.list_available_spks()
            elif name == 'add_zero_shot_spk':
                result = self.cosyvoice.add_zero_shot_spk(**kwargs)
            elif name == 'save_spkinfo':
                self.cosyvoice.save_spkinfo()
                result = None
            elif name == 'inference_sft':
                result = self.cosyvoice.inference_sft(**kwargs)
            elif name == 'inference_zero_shot':
                result = self.cosyvoice.inference_zero_shot(**kwargs)
            elif name == 'inference_cross_lingual':
                result = self.cosyvoice.inference_cross_lingual(**kwargs)
            elif name == 'inference_instruct':
                result = self.cosyvoice.inference_instruct(**kwargs)
            elif name == 'inference_instruct2':
                result = self.cosyvoice.inference_instruct2(**kwargs)
            elif name == 'inference_vc':
                result = self.cosyvoice.inference_vc(**kwargs)
            else:
                raise ValueError(f"Method {name} not recognized.")

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
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('list_available_spks')
adapter_additional_data['functions'].append('add_zero_shot_spk')
adapter_additional_data['functions'].append('save_spkinfo')
adapter_additional_data['functions'].append('inference_sft')
adapter_additional_data['functions'].append('inference_zero_shot')
adapter_additional_data['functions'].append('inference_cross_lingual')
adapter_additional_data['functions'].append('inference_instruct')
adapter_additional_data['functions'].append('inference_instruct2')
adapter_additional_data['functions'].append('inference_vc')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
