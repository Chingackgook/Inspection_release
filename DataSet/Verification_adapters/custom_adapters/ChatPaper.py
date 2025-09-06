# ChatPaper 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'ChatPaper/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/ChatPaper')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/ChatPaper')

# 可以在此位置后添加导包部分代码
from chat_paper import *


import argparse
import base64
import configparser
import datetime
import json
import os
import re
from collections import namedtuple
import arxiv
import numpy as np
import openai
import requests
import tenacity
import tiktoken
import fitz, io, os
from PIL import Image
import sys
import sys
import sys
import time


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.reader_obj = None  # Reader类的接口对象

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'Reader':
                # 创建Reader接口对象
                self.reader_obj = Reader(**kwargs)
                self.result.interface_return = self.reader_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认Reader接口对象
                self.reader_obj = Reader(**kwargs)
                self.result.interface_return = self.reader_obj
            
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
            if dispatch_key == 'get_arxiv':
                self.result.interface_return = self.reader_obj.get_arxiv(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'filter_arxiv':
                self.result.interface_return = self.reader_obj.filter_arxiv(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'validateTitle':
                self.result.interface_return = self.reader_obj.validateTitle(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'download_pdf':
                self.result.interface_return = self.reader_obj.download_pdf(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'try_download_pdf':
                self.result.interface_return = self.reader_obj.try_download_pdf(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'upload_gitee':
                self.result.interface_return = self.reader_obj.upload_gitee(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'summary_with_chat':
                self.result.interface_return = self.reader_obj.summary_with_chat(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'chat_conclusion':
                self.result.interface_return = self.reader_obj.chat_conclusion(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'chat_method':
                self.result.interface_return = self.reader_obj.chat_method(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'chat_summary':
                self.result.interface_return = self.reader_obj.chat_summary(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'export_to_markdown':
                self.result.interface_return = self.reader_obj.export_to_markdown(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'show_info':
                self.result.interface_return = self.reader_obj.show_info()
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
