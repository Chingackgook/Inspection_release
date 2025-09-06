# ChatDev 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'ChatDev/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/ChatDev')

# 可以在此位置后添加导包部分代码
import argparse
import logging
import os
import sys
from camel.typing import ModelType
from chatdev.chat_chain import ChatChain
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message import FunctionCall


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        try:
            # 假设我们需要加载 ChatChain 类
            self.chat_chain = ChatChain(**kwargs)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.chat_chain
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
        try:
            if name == 'make_recruitment':
                self.chat_chain.make_recruitment()
                self.result.set_result(
                    fuc_name='make_recruitment',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'execute_step':
                phase_item = kwargs.get('phase_item')
                self.chat_chain.execute_step(phase_item)
                self.result.set_result(
                    fuc_name='execute_step',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'execute_chain':
                self.chat_chain.execute_chain()
                self.result.set_result(
                    fuc_name='execute_chain',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'get_logfilepath':
                start_time, log_filepath = self.chat_chain.get_logfilepath()
                self.result.set_result(
                    fuc_name='get_logfilepath',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data={'start_time': start_time, 'log_filepath': log_filepath},
                    interface_return=(start_time, log_filepath)
                )
            elif name == 'pre_processing':
                self.chat_chain.pre_processing()
                self.result.set_result(
                    fuc_name='pre_processing',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'post_processing':
                self.chat_chain.post_processing()
                self.result.set_result(
                    fuc_name='post_processing',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'self_task_improve':
                task_prompt = kwargs.get('task_prompt')
                revised_task_prompt = self.chat_chain.self_task_improve(task_prompt)
                self.result.set_result(
                    fuc_name='self_task_improve',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=revised_task_prompt,
                    interface_return=revised_task_prompt
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
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('make_recruitment')
adapter_additional_data['functions'].append('execute_step')
adapter_additional_data['functions'].append('execute_chain')
adapter_additional_data['functions'].append('get_logfilepath')
adapter_additional_data['functions'].append('pre_processing')
adapter_additional_data['functions'].append('post_processing')
adapter_additional_data['functions'].append('self_task_improve')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
