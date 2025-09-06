# chatTTS 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'chatTTS/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/ChatTTS')
# 以上是自动生成的代码，请勿修改



import ChatTTS
from abc import ABC, abstractmethod
from typing import Any



class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.chat = None

    def create_interface_objects(self, **kwargs):
        try:
            if self.chat is not None:
                self.result.set_result('create_interface_objects', False, 'Model already loaded', False, '', None, None)
            self.chat = ChatTTS.Chat()
            self.chat.load(**kwargs)
            self.result.set_result('create_interface_objects', True, '', False, '', None, None)
        except Exception as e:
            self.result.set_result('create_interface_objects', False, str(e), False, '', None, None)

    def run(self, name: str, **kwargs):
        try:
            if name == 'download_models':
                output = self.chat.download_models(**kwargs)
            elif name == 'has_loaded':
                output = self.chat.has_loaded(**kwargs)
            elif name == 'infer':
                output = self.chat.infer(**kwargs)
            elif name == 'unload':
                output = self.chat.unload()
            elif name == 'sample_random_speaker':
                output = self.chat.sample_random_speaker()
            elif name == 'sample_audio_speaker':
                output = self.chat.sample_audio_speaker(**kwargs)
            elif name == 'interrupt':
                output = self.chat.interrupt()
            elif name.startswith('Chat_'):
                class_method = name.split('_', 1)[1]
                output = getattr(self.chat, class_method)(**kwargs)
            else:
                raise ValueError(f"Unknown method name: {name}")

            self.result.set_result(name, True, '', False, '', output, output)
        except Exception as e:
            self.result.set_result(name, False, str(e), False, '', None, None)

# 这是一个外层调用示例，供参考
# adapter = CustomAdapter()
# adapter.create_interface_objects(source='local', custom_path='模型路径')
# adapter.run(name='infer', text='Hello')
# print(adapter.result)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('download_models')
adapter_addtional_data['functions'].append('has_loaded')
adapter_addtional_data['functions'].append('infer')
adapter_addtional_data['functions'].append('unload')
adapter_addtional_data['functions'].append('sample_random_speaker')
adapter_addtional_data['functions'].append('sample_audio_speaker')
adapter_addtional_data['functions'].append('interrupt')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
