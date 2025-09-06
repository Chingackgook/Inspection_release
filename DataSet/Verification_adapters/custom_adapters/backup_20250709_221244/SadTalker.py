# SadTalker 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'SadTalker/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/SadTalker')

# 可以在此位置后添加导包部分代码
from glob import glob
import shutil
import torch
from time import strftime
import os
import sys
import time
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff,load_cpk #此处添加load_cpk导入
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from src.face3d.visualize import gen_composed_video


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.audio2coeff_obj = None  # Audio2Coeff 的接口对象
        try:
            if interface_class_name == 'Audio2Coeff':
                # 创建接口对象
                self.audio2coeff_obj = Audio2Coeff(**kwargs)
                self.result.interface_return = self.audio2coeff_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象
                self.audio2coeff_obj = Audio2Coeff(**kwargs)
                self.result.interface_return = self.audio2coeff_obj
            
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 创建接口对象失败: {e}")

    def run(self, name: str, **kwargs):
        try:
            if name == 'load_cpk':
                # 执行独立函数
                self.result.interface_return = load_cpk(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'generate':
                # 执行 Audio2Coeff 的 generate 方法
                self.result.interface_return = self.audio2coeff_obj.generate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'using_refpose':
                # 执行 Audio2Coeff 的 using_refpose 方法
                self.result.interface_return = self.audio2coeff_obj.using_refpose(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            else:
                raise ValueError(f"Unknown method name: {name}")

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行方法 {name} 失败: {e}")

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('Audio2Coeff')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('load_cpk')
adapter_additional_data['functions'].append('generate')
adapter_additional_data['functions'].append('using_refpose')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
