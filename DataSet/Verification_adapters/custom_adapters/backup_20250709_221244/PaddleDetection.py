from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# PaddleDetection 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'PaddleDetection/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/PaddleDetection')
# 可以在此位置后添加导包部分代码
import os
import sys
import warnings
import glob
import ast
import paddle
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config
from ppdet.core.workspace import merge_config
from ppdet.engine import Trainer
from ppdet.engine import Trainer_ARSL
from ppdet.utils.check import check_gpu
from ppdet.utils.check import check_npu
from ppdet.utils.check import check_xpu
from ppdet.utils.check import check_mlu
from ppdet.utils.check import check_gcu
from ppdet.utils.check import check_version
from ppdet.utils.check import check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.cli import merge_args
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.trainer = None

    def create_interface_objects(self, cfg, mode='train'):
        try:
            self.trainer = Trainer(cfg, mode)
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
            if name == 'register_callbacks':
                self.trainer.register_callbacks(**kwargs)
            elif name == 'register_metrics':
                self.trainer.register_metrics(**kwargs)
            elif name == 'load_weights':
                self.trainer.load_weights(**kwargs)
            elif name == 'load_weights_sde':
                self.trainer.load_weights_sde(**kwargs)
            elif name == 'resume_weights':
                self.trainer.resume_weights(**kwargs)
            elif name == 'train':
                self.trainer.train(**kwargs)
            elif name == 'evaluate':
                self.trainer.evaluate()
            elif name == 'evaluate_slice':
                self.trainer.evaluate_slice(**kwargs)
            elif name == 'slice_predict':
                result = self.trainer.slice_predict(**kwargs)
                self.result.interface_return = result
            elif name == 'predict':
                result = self.trainer.predict(**kwargs)
                self.result.interface_return = result
            elif name == 'export':
                self.trainer.export(**kwargs)
            elif name == 'post_quant':
                self.trainer.post_quant(**kwargs)
            elif name == 'parse_mot_images':
                result = self.trainer.parse_mot_images(**kwargs)
                self.result.interface_return = result
            elif name == 'predict_culane':
                result = self.trainer.predict_culane(**kwargs)
                self.result.interface_return = result
            elif name == 'reset_norm_param_attr':
                result = self.trainer.reset_norm_param_attr(**kwargs)
                self.result.interface_return = result
            elif name == 'setup_metrics_for_loader':
                result = self.trainer.setup_metrics_for_loader()
                self.result.interface_return = result
            elif name == 'deep_pin':
                result = self.trainer.deep_pin(**kwargs)
                self.result.interface_return = result
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.result.interface_return
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
adapter_additional_data['functions'].append('register_callbacks')
adapter_additional_data['functions'].append('register_metrics')
adapter_additional_data['functions'].append('load_weights')
adapter_additional_data['functions'].append('load_weights_sde')
adapter_additional_data['functions'].append('resume_weights')
adapter_additional_data['functions'].append('train')
adapter_additional_data['functions'].append('evaluate')
adapter_additional_data['functions'].append('evaluate_slice')
adapter_additional_data['functions'].append('slice_predict')
adapter_additional_data['functions'].append('predict')
adapter_additional_data['functions'].append('export')
adapter_additional_data['functions'].append('post_quant')
adapter_additional_data['functions'].append('parse_mot_images')
adapter_additional_data['functions'].append('predict_culane')
adapter_additional_data['functions'].append('reset_norm_param_attr')
adapter_additional_data['functions'].append('setup_metrics_for_loader')
adapter_additional_data['functions'].append('deep_pin')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
