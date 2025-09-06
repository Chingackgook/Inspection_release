# Open_Sora_Plan 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Open_Sora_Plan/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/Open-Sora-Plan')

# 可以在此位置后添加导包部分代码
import gradio as gr
import os
import torch
from einops import rearrange
import torch.distributed as dist
from torchvision.utils import save_image
import imageio
import math
import argparse
import random
import numpy as np
import string
from opensora.sample.caption_refiner import OpenSoraCaptionRefiner
from opensora.utils.sample_utils import prepare_pipeline
from opensora.utils.sample_utils import save_video_grid
from opensora.utils.sample_utils import init_gpu_env
from gradio_utils import *
from opensora.sample.VEnhancer.enhance_a_video import VEnhancer


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # 不需要初始化任何接口类的对象
        pass

    def run(self, name: str, **kwargs):
        try:
            if name == 'generate':
                # 执行 generate 函数
                self.result.interface_return = generate(**kwargs)
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
adapter_additional_data['functions'].append('generate')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
