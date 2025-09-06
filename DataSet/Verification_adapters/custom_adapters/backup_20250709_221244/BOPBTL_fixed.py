# BOPBTL_fixed 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'BOPBTL_fixed/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/Bringing-Old-Photos-Back-to-Life/Global')
# 以上是自动生成的代码，请勿修改


from options.test_options import TestOptions
from test_model import data_transforms,data_transforms_rgb_old,irregular_hole_synthesize
from models.mapping_model import Pix2PixHDModel_Mapping
from abc import ABC, abstractmethod
from typing import Any

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_interface_objects(self, **kwargs):
        """
        加载模型并初始化
        """
        try:
            self.model = Pix2PixHDModel_Mapping()
            self.model.initialize(kwargs.get('opt'))
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
        """
        执行具体智能模块的入口方法
        """
        try:
            if name == 'name':
                self.result.interface_return = self.model.name()
                self.result.set_result(
                    fuc_name='name',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'init_loss_filter':
                self.result.interface_return = self.model.init_loss_filter(**kwargs)
                self.result.set_result(
                    fuc_name='init_loss_filter',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'initialize':
                self.model.initialize(**kwargs)
                self.result.set_result(
                    fuc_name='initialize',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'encode_input':
                self.result.interface_return = self.model.encode_input(**kwargs)
                self.result.set_result(
                    fuc_name='encode_input',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'discriminate':
                self.result.interface_return = self.model.discriminate(**kwargs)
                self.result.set_result(
                    fuc_name='discriminate',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'forward':
                self.result.interface_return = self.model.forward(**kwargs)
                self.result.set_result(
                    fuc_name='forward',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=self.result.interface_return
                )
            elif name == 'inference':
                print('Running inference with kwargs:', kwargs)
                self.result.interface_return = self.model.inference(**kwargs)
                self.result.set_result(
                    fuc_name='inference',
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
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('name')
adapter_addtional_data['functions'].append('init_loss_filter')
adapter_addtional_data['functions'].append('initialize')
adapter_addtional_data['functions'].append('encode_input')
adapter_addtional_data['functions'].append('discriminate')
adapter_addtional_data['functions'].append('forward')
adapter_addtional_data['functions'].append('inference')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
