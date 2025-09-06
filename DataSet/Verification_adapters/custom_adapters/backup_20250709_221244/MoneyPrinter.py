# MoneyPrinter 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'MoneyPrinter/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
#修改前
#sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/MoneyPrinter')

# 修改后
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/MoneyPrinter/Backend')

# 可以在此位置后添加导包部分代码
import os
from utils import *
from dotenv import load_dotenv
from gpt import *
from video import *
from search import *
from uuid import uuid4
from tiktokvoice import *
from flask_cors import CORS
from termcolor import colored
from youtube import upload_video
from apiclient.errors import HttpError
from flask import Flask
from flask import request
from flask import jsonify
from moviepy.config import change_settings


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        # 不需要初始化任何接口类的对象
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_file = False
        self.result.file_path = ''
    
    def run(self, name: str, **kwargs):
        try:
            if name == 'save_video':
                self.result.interface_return = save_video(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'generate_subtitles':
                self.result.interface_return = generate_subtitles(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'combine_videos':
                self.result.interface_return = combine_videos(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'generate_video':
                self.result.interface_return = generate_video(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'convert_to_srt_time_format':
                self.result.interface_return = convert_to_srt_time_format(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'equalize_subtitles':
                equalize_subtitles(**kwargs)
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
adapter_additional_data['functions'].append('save_video')
adapter_additional_data['functions'].append('generate_subtitles')
adapter_additional_data['functions'].append('combine_videos')
adapter_additional_data['functions'].append('generate_video')
adapter_additional_data['functions'].append('convert_to_srt_time_format')
adapter_additional_data['functions'].append('equalize_subtitles')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
