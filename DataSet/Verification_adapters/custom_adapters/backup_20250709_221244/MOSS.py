# MOSS 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'MOSS/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/MOSS')

# 可以在此位置后添加导包部分代码
import time
import statistics
import json
import re
from typing import Union
from typing import List
from typing import Tuple
from typing import Optional
from typing import Dict
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights
from accelerate import load_checkpoint_and_dispatch
# 删除
# from transformers import MossForCausalLM
# from transformers import MossTokenizer
# from transformers import MossConfig
import os
from models.modeling_moss import MossForCausalLM
from models.tokenization_moss import MossTokenizer
from models.configuration_moss import MossConfig


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # MossAttention
        self.class2_obj = None  # MossMLP
        self.class3_obj = None  # MossBlock
        self.class4_obj = None  # MossModel
        self.class5_obj = None  # MossForCausalLM
        
        try:
            if interface_class_name == 'MossAttention':
                self.class1_obj = MossAttention(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'MossMLP':
                self.class2_obj = MossMLP(**kwargs)
                self.result.interface_return = self.class2_obj
            elif interface_class_name == 'MossBlock':
                self.class3_obj = MossBlock(**kwargs)
                self.result.interface_return = self.class3_obj
            elif interface_class_name == 'MossModel':
                self.class4_obj = MossModel(**kwargs)
                self.result.interface_return = self.class4_obj
            elif interface_class_name == 'MossForCausalLM':
                self.class5_obj = MossForCausalLM(**kwargs)
                self.result.interface_return = self.class5_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象（假设只有一个接口类）
                self.class4_obj = MossModel(**kwargs)
                self.result.interface_return = self.class4_obj
            
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 创建接口对象失败: {e}")

    def run(self, name: str, **kwargs):
        try:
            if name == 'create_sinusoidal_positions':
                self.result.interface_return = create_sinusoidal_positions(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'rotate_every_two':
                self.result.interface_return = rotate_every_two(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'apply_rotary_pos_emb':
                self.result.interface_return = apply_rotary_pos_emb(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'noop':
                self.result.interface_return = noop(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'create_custom_forward':
                self.result.interface_return = create_custom_forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'custom_forward':
                self.result.interface_return = custom_forward(*kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossAttention_forward':
                self.result.interface_return = self.class1_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossMLP_forward':
                self.result.interface_return = self.class2_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossBlock_forward':
                self.result.interface_return = self.class3_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossModel_get_input_embeddings':
                self.result.interface_return = self.class4_obj.get_input_embeddings()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossModel_set_input_embeddings':
                self.result.interface_return = self.class4_obj.set_input_embeddings(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossModel_forward':
                self.result.interface_return = self.class4_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossForCausalLM_get_output_embeddings':
                self.result.interface_return = self.class5_obj.get_output_embeddings()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossForCausalLM_set_output_embeddings':
                self.result.interface_return = self.class5_obj.set_output_embeddings(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossForCausalLM_prepare_inputs_for_generation':
                self.result.interface_return = self.class5_obj.prepare_inputs_for_generation(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'MossForCausalLM_forward':
                self.result.interface_return = self.class5_obj.forward(**kwargs)
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
adapter_additional_data['functions'].append('MossAttention')
adapter_additional_data['functions'].append('MossMLP')
adapter_additional_data['functions'].append('MossBlock')
adapter_additional_data['functions'].append('MossModel')
adapter_additional_data['functions'].append('MossForCausalLM')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('create_sinusoidal_positions')
adapter_additional_data['functions'].append('rotate_every_two')
adapter_additional_data['functions'].append('apply_rotary_pos_emb')
adapter_additional_data['functions'].append('noop')
adapter_additional_data['functions'].append('create_custom_forward')
adapter_additional_data['functions'].append('custom_forward')
adapter_additional_data['functions'].append('MossAttention_forward')
adapter_additional_data['functions'].append('MossMLP_forward')
adapter_additional_data['functions'].append('MossBlock_forward')
adapter_additional_data['functions'].append('MossModel_get_input_embeddings')
adapter_additional_data['functions'].append('MossModel_set_input_embeddings')
adapter_additional_data['functions'].append('MossModel_forward')
adapter_additional_data['functions'].append('MossForCausalLM_get_output_embeddings')
adapter_additional_data['functions'].append('MossForCausalLM_set_output_embeddings')
adapter_additional_data['functions'].append('MossForCausalLM_prepare_inputs_for_generation')
adapter_additional_data['functions'].append('MossForCausalLM_forward')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
