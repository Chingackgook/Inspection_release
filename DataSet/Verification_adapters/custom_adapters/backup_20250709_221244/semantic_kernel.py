# semantic_kernel 
from calendar import c
from curses import raw
from types import coroutine

from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'semantic-kernel/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/semantic-kernel/python')
# 以上是自动生成的代码，请勿修改


from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.contents import ChatMessageContent,FunctionCallContent,FunctionResultContent

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.agent = None

    def create_interface_objects(self, **kwargs):
        """
        初始化 ChatCompletionAgent 实例，并将其存储在 self.agent 中。
        """
        self.agent = ChatCompletionAgent(
            **kwargs
        )
        self.result.set_result(
            fuc_name='create_interface_objects',
            is_success=True,
            fail_reason='',
            is_file=False,
            file_path='',
            except_data=None,
            interface_return=self.agent
        )


    def run(self, name: str, **kwargs):
        """
        根据 name 执行对应的方法，并将结果存储在 self.result 中。
        """
        if name == 'create_channel':
            chat_history = kwargs.get('chat_history', None)
            thread_id = kwargs.get('thread_id', None)
            channel = self.agent.create_channel(chat_history=chat_history, thread_id=thread_id)
            self.result.set_result(
                fuc_name='create_channel',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=channel
            )
        elif name == 'get_response':
            messages = kwargs.get('messages', None)
            thread = kwargs.get('thread', None)
            response = self.agent.get_response(messages=messages, thread=thread)

            self.result.set_result(
                fuc_name='get_response',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=response,
                interface_return=response
            )
        elif name == 'invoke':
            messages = kwargs.get('messages', None)
            thread = kwargs.get('thread', None)
            for item in self.agent.invoke(messages=messages, thread=thread):
                self.result.set_result(
                    fuc_name='invoke',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=item.message.content,
                    interface_return=item
                )
        elif name == 'invoke_stream':
            messages = kwargs.get('messages', None)
            thread = kwargs.get('thread', None)
            for item in self.agent.invoke_stream(messages=messages, thread=thread):
                self.result.set_result(
                    fuc_name='invoke_stream',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=item.message.content,
                    interface_return=item
                )
        elif name == '_inner_invoke':
            thread = kwargs.get('thread', None)
            history = kwargs.get('history', None)
            on_intermediate_message = kwargs.get('on_intermediate_message', None)
            for item in self.agent._inner_invoke(thread=thread, history=history, on_intermediate_message=on_intermediate_message):
                self.result.set_result(
                    fuc_name='_inner_invoke',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=item,
                    interface_return=item
                )
        elif name == '_prepare_agent_chat_history':
            history = kwargs.get('history', None)
            kernel = kwargs.get('kernel', None)
            arguments = kwargs.get('arguments', None)
            prepared_history = self.agent._prepare_agent_chat_history(history=history, kernel=kernel, arguments=arguments)
            self.result.set_result(
                fuc_name='_prepare_agent_chat_history',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=prepared_history
            )
        elif name == '_get_chat_completion_service_and_settings':
            kernel = kwargs.get('kernel', None)
            arguments = kwargs.get('arguments', None)
            service, settings = self.agent._get_chat_completion_service_and_settings(kernel=kernel, arguments=arguments)
            self.result.set_result(
                fuc_name='_get_chat_completion_service_and_settings',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=(service, settings),
                interface_return=(service, settings)
            )
        elif name == '_capture_mutated_messages':
            agent_chat_history = kwargs.get('agent_chat_history', None)
            start = kwargs.get('start', 0)
            thread = kwargs.get('thread', None)
            on_intermediate_message = kwargs.get('on_intermediate_message', None)
            self.agent._capture_mutated_messages(agent_chat_history=agent_chat_history, start=start, thread=thread, on_intermediate_message=on_intermediate_message)
            self.result.set_result(
                fuc_name='_capture_mutated_messages',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
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

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('create_channel')
adapter_additional_data['functions'].append('get_response')
adapter_additional_data['functions'].append('invoke')
adapter_additional_data['functions'].append('invoke_stream')
adapter_additional_data['functions'].append('_inner_invoke')
adapter_additional_data['functions'].append('_prepare_agent_chat_history')
adapter_additional_data['functions'].append('_get_chat_completion_service_and_settings')
adapter_additional_data['functions'].append('_capture_mutated_messages')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
