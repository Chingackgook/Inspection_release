# ChatPaper 
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
from Inspection import ENV_BASE
import os
ENV_DIR = ENV_BASE + 'ChatPaper/'
import sys
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/ChatPaper')
# 以上是自动生成的代码，请勿修改


from abc import ABC
from typing import Any, Dict, List
import argparse
import arxiv
from chat_paper import Paper
from chat_paper import Reader

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.reader = None

    def create_interface_objects(self, key_word: str, query: str, filter_keys: str, root_path: str = './', 
                   gitee_key: str = '', sort=arxiv.SortCriterion.SubmittedDate, user_name: str = 'default', 
                   args: argparse.Namespace = None):
        """
        初始化Reader类的实例
        """
        args = argparse.Namespace()
        args.language = 'zh'
        args.file_format = 'md'
        args.save_image = False
        self.reader = Reader(key_word=key_word, query=query, filter_keys=filter_keys, 
                                root_path=root_path, gitee_key=gitee_key, sort=sort, 
                                user_name=user_name, args=args)
        self.result.set_result(fuc_name='create_interface_objects', is_success=True, fail_reason='', 
                                is_file=False, file_path='', except_data=None, 
                                interface_return=self.reader)
    def run(self, name: str, **kwargs):
        """
        执行具体智能模块的入口方法
        """
        try:
            if name == 'summary_with_chat':
                self.reader.summary_with_chat(**kwargs)
                self.result.set_result(fuc_name=name, is_success=True, fail_reason='', 
                                       is_file=False, file_path='', except_data=None, 
                                       interface_return=self.reader)
            elif name == 'chat_conclusion':
                conclusion = self.reader.chat_conclusion(**kwargs)
                self.result.set_result(fuc_name=name, is_success=True, fail_reason='', 
                                       is_file=False, file_path='', except_data=conclusion, 
                                       interface_return=conclusion)
            elif name == 'chat_method':
                method = self.reader.chat_method(**kwargs)
                self.result.set_result(fuc_name=name, is_success=True, fail_reason='', 
                                       is_file=False, file_path='', except_data=method, 
                                       interface_return=method)
            elif name == 'chat_summary':
                summary = self.reader.chat_summary(**kwargs)
                self.result.set_result(fuc_name=name, is_success=True, fail_reason='', 
                                       is_file=False, file_path='', except_data=summary, 
                                       interface_return=summary)
            else:
                self.result.set_result(fuc_name=name, is_success=False, 
                                       fail_reason='Function not found', is_file=False, 
                                       file_path='', except_data=None, interface_return=None)
        except Exception as e:
            self.result.set_result(fuc_name=name, is_success=False, 
                                   fail_reason=str(e), is_file=False, file_path='', 
                                   except_data=None, interface_return=None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('summary_with_chat')
adapter_addtional_data['functions'].append('chat_conclusion')
adapter_addtional_data['functions'].append('chat_method')
adapter_addtional_data['functions'].append('chat_summary')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)


if __name__ == "__main__":
    # 测试代码
    adapter = CustomAdapter()
    adapter.create_interface_objects(key_word='Machine Learning', query='deep learning', filter_keys='neural network', root_path='./')
    reader = adapter.reader
    search_results = reader.filter_arxiv(max_results=20)
    print("筛选后的arXiv结果：", search_results)
    for index, result in enumerate(search_results):
        print(index, result)