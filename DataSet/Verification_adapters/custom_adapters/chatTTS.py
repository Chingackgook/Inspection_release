# chatTTS 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'chatTTS/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/ChatTTS')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/ChatTTS')

# 可以在此位置后添加导包部分代码
from ChatTTS.core import *


import os
import sys
import logging
import ChatTTS
from tools.logger import get_logger
import sys


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.chat_obj = None  # Chat 类的接口对象
        try:
            if interface_class_name == 'Chat':
                # 创建接口对象
                self.chat_obj = Chat(**kwargs)
                self.result.interface_return = self.chat_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象
                self.chat_obj = Chat(**kwargs)
                self.result.interface_return = self.chat_obj
            
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
            if dispatch_key == 'has_loaded':
                self.result.interface_return = self.chat_obj.has_loaded(**kwargs)
            elif dispatch_key == 'download_models':
                self.result.interface_return = self.chat_obj.download_models(**kwargs)
            elif dispatch_key == 'load':
                self.result.interface_return = self.chat_obj.load(**kwargs)
            elif dispatch_key == 'unload':
                self.chat_obj.unload()
                self.result.interface_return = None
            elif dispatch_key == 'sample_random_speaker':
                self.result.interface_return = self.chat_obj.sample_random_speaker()
            elif dispatch_key == 'sample_audio_speaker':
                self.result.interface_return = self.chat_obj.sample_audio_speaker(**kwargs)
            elif dispatch_key == 'infer':
                self.result.interface_return = self.chat_obj.infer(**kwargs)
            elif dispatch_key == 'interrupt':
                self.chat_obj.interrupt()
                self.result.interface_return = None
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = dispatch_key

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行接口 {dispatch_key} 失败: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
