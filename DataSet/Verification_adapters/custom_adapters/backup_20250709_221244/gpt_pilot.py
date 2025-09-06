# gpt_pilot 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gpt_pilot/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/gpt-pilot')
# 以上是自动生成的代码，请勿修改


from typing import Any, Optional, Callable, Tuple
from abc import ABC, abstractmethod
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.client = None

    def create_interface_objects(self, **kwargs):
        """
        初始化 BaseLLMClient，并将其存储在 self.client 中。
        """
        try:
            config = kwargs.get('config')
            self.client = BaseLLMClient(config)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.client
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
        根据 name 执行对应的方法，并将结果存储在 self.result 中。
        """
        try:
            if name == '_make_request':
                convo = kwargs.get('convo')
                temperature = kwargs.get('temperature')
                json_mode = kwargs.get('json_mode', False)
                response, input_tokens, output_tokens = self.client._make_request(convo, temperature, json_mode)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=(response, input_tokens, output_tokens)
                )
            elif name == 'api_check':
                success = self.client.api_check()
                self.result.set_result(
                    fuc_name=name,
                    is_success=success,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=success
                )
            elif name == '_adapt_messages':
                convo = kwargs.get('convo')
                adapted_messages = self.client._adapt_messages(convo)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=adapted_messages,
                    interface_return=adapted_messages
                )
            elif name == '__call__':
                convo = kwargs.get('convo')
                temperature = kwargs.get('temperature')
                parser = kwargs.get('parser')
                max_retries = kwargs.get('max_retries', 3)
                json_mode = kwargs.get('json_mode', False)
                response, request_log = self.client(convo, temperature=temperature, parser=parser, max_retries=max_retries, json_mode=json_mode)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=(response, request_log)
                )
            elif name == 'rate_limit_sleep':
                err = kwargs.get('err')
                wait_time = self.client.rate_limit_sleep(err)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=wait_time,
                    interface_return=wait_time
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
adapter_addtional_data['functions'].append('_make_request')
adapter_addtional_data['functions'].append('api_check')
adapter_addtional_data['functions'].append('_adapt_messages')
adapter_addtional_data['functions'].append('__call__')
adapter_addtional_data['functions'].append('rate_limit_sleep')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
