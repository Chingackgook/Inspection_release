# gpt_academic 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gpt_academic/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/gpt_academic')

# 可以在此位置后添加导包部分代码
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import update_ui, get_conf, trimmed_format_exc, get_max_token
from request_llms.bridge_all import predict_no_ui_long_connection



# DeadCodeFront end

from typing import Any, Dict, List
from abc import ABC, abstractmethod

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.chatbot = None  # Placeholder for chatbot object
        self.history = []    # Placeholder for history list

    def create_interface_objects(self, **kwargs):
        # Load the model and store the result
        try:
            # Assuming model loading logic here
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
        if name == 'request_gpt_model_in_new_thread_with_ui_alive':
            try:
                future = request_gpt_model_in_new_thread_with_ui_alive(
                    inputs=kwargs.get('inputs'),
                    inputs_show_user=kwargs.get('inputs_show_user'),
                    llm_kwargs=kwargs.get('llm_kwargs'),
                    chatbot=kwargs.get('chatbot'),
                    history=kwargs.get('history'),
                    sys_prompt=kwargs.get('sys_prompt'),
                    refresh_interval=kwargs.get('refresh_interval', 0.2),
                    handle_token_exceed=kwargs.get('handle_token_exceed', True),
                    retry_times_at_unknown_error=kwargs.get('retry_times_at_unknown_error', 2)
                )
                result_data = []
                for item in future:
                    result_data.append(item)
                    break #只取第一个结果
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=result_data,
                    interface_return=future
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
        
        elif name == 'predict_no_ui_long_connection':
            try:
                result = predict_no_ui_long_connection(
                    inputs=kwargs.get('inputs'),
                    llm_kwargs=kwargs.get('llm_kwargs'),
                    history=self.history,
                    sys_prompt=kwargs.get('sys_prompt'),
                    observe_window=kwargs.get('observe_window')
                )
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=result,
                    interface_return=result
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
        
        elif name == 'update_ui':
            try:
                result = update_ui(
                    **kwargs
                )
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=result
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
        
        elif name == 'get_conf':
            try:
                value = get_conf(kwargs.get('key'))
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=value,
                    interface_return=value
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
        
        elif name == 'trimmed_format_exc':
            try:
                exc_info = trimmed_format_exc()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=exc_info,
                    interface_return=exc_info
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
        
        elif name == 'get_max_token':
            try:
                max_token = get_max_token(kwargs.get('llm_kwargs'))
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=max_token,
                    interface_return=max_token
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
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('request_gpt_model_in_new_thread_with_ui_alive')
adapter_additional_data['functions'].append('predict_no_ui_long_connection')
adapter_additional_data['functions'].append('update_ui')
adapter_additional_data['functions'].append('get_conf')
adapter_additional_data['functions'].append('trimmed_format_exc')
adapter_additional_data['functions'].append('get_max_token')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
