# BOPBTL_fixed 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'BOPBTL_fixed/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/Bringing-Old-Photos-Back-to-Life/Global')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/Bringing-Old-Photos-Back-to-Life/Global')

# 可以在此位置后添加导包部分代码
from models.mapping_model import *


import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping
import util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
import cv2


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.mapping_model_obj = None
        self.pix2pixhd_model_mapping_obj = None
        
        try:
            if interface_class_name == 'Mapping_Model':
                self.mapping_model_obj = Mapping_Model(**kwargs)
                self.result.interface_return = self.mapping_model_obj
            elif interface_class_name == 'Pix2PixHDModel_Mapping':
                self.pix2pixhd_model_mapping_obj = Pix2PixHDModel_Mapping(**kwargs)
                self.result.interface_return = self.pix2pixhd_model_mapping_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象（假设只有一个接口类）
                self.pix2pixhd_model_mapping_obj = Pix2PixHDModel_Mapping(**kwargs)
                self.result.interface_return = self.pix2pixhd_model_mapping_obj
            
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

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'name':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.name()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'init_loss_filter':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.init_loss_filter(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'loss_filter':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.loss_filter(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'initialize':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.initialize(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'encode_input':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.encode_input(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'discriminate':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.discriminate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'forward':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'inference':
                self.result.interface_return = self.pix2pixhd_model_mapping_obj.inference(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'Mapping_Model_forward':
                self.result.interface_return = self.mapping_model_obj.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行接口 {dispatch_key} 失败: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
