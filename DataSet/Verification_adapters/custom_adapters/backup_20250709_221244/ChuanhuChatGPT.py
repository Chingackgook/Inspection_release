# ChuanhuChatGPT 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'ChuanhuChatGPT/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/ChuanhuChatGPT')

# 可以在此位置后添加导包部分代码
import os
from modules.index_func import *
from modules.presets import *
from modules.models.OpenAIVision import OpenAIVisionClient

# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None

    def create_interface_objects(self, model_name: str, api_key: str, user_name: str = ""):
        """
        初始化 OpenAIVisionClient 实例
        """
        try:
            self.model = OpenAIVisionClient(model_name=model_name, api_key=api_key, user_name=user_name)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.model
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
        执行具体智能模块的入口方法
        """
        try:
            if name == 'get_answer_stream_iter':
                response = self.model.get_answer_stream_iter()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=response
                )
            elif name == 'get_answer_at_once':
                content, total_token_count = self.model.get_answer_at_once()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data={'content': content, 'total_token_count': total_token_count},
                    interface_return=(content, total_token_count)
                )
            elif name == 'predict':
                response = self.model.predict(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=response
                )
            elif name == 'retry':
                response = self.model.retry(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=response,
                    interface_return=response
                )
            elif name == 'count_token':
                token_count = self.model.count_token(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=token_count,
                    interface_return=token_count
                )
            elif name == 'count_image_tokens':
                image_token_count = self.model.count_image_tokens(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=image_token_count,
                    interface_return=image_token_count
                )
            elif name == 'billing_info':
                billing_info = self.model.billing_info()
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=billing_info,
                    interface_return=billing_info
                )
            elif name == 'set_key':
                success = self.model.set_key(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=success,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=success
                )
            elif name == 'auto_name_chat_history':
                updated_status = self.model.auto_name_chat_history(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=updated_status,
                    interface_return=updated_status
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
adapter_additional_data['functions'].append('get_answer_stream_iter')
adapter_additional_data['functions'].append('get_answer_at_once')
adapter_additional_data['functions'].append('count_token')
adapter_additional_data['functions'].append('count_image_tokens')
adapter_additional_data['functions'].append('billing_info')
adapter_additional_data['functions'].append('_get_gpt4v_style_history')
adapter_additional_data['functions'].append('_get_response')
adapter_additional_data['functions'].append('_refresh_header')
adapter_additional_data['functions'].append('_get_billing_data')
adapter_additional_data['functions'].append('_decode_chat_response')
adapter_additional_data['functions'].append('set_key')
adapter_additional_data['functions'].append('_single_query_at_once')
adapter_additional_data['functions'].append('auto_name_chat_history')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
