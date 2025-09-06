# gpt_engineer 
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
from Inspection import TEST_DATA_PATH
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gpt_engineer/'
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/gpt_engineer')
# 以上是自动生成的代码，请勿修改


from dotenv import load_dotenv
from typing import Any, List, Optional

# 这里假设AI和ClipboardAI已经被定义并可以导入
from gpt_engineer.core.ai import AI, ClipboardAI
from gpt_engineer.core.ai import AI
from gpt_engineer.core.base_execution_env import BaseExecutionEnv
from gpt_engineer.core.base_memory import BaseMemory
from gpt_engineer.core.chat_to_files import apply_diffs, chat_to_files_dict, parse_diffs
from gpt_engineer.core.default.constants import MAX_EDIT_REFINEMENT_STEPS
from gpt_engineer.core.default.paths import (
    CODE_GEN_LOG_FILE,
    DEBUG_LOG_FILE,
    DIFF_LOG_FILE,
    ENTRYPOINT_FILE,
    ENTRYPOINT_LOG_FILE,
    IMPROVE_LOG_FILE,
)
from gpt_engineer.core.files_dict import FilesDict, file_to_lines_dict
from gpt_engineer.core.preprompts_holder import PrepromptsHolder
from gpt_engineer.core.prompt import Prompt
from gpt_engineer.core.default.steps import curr_fn, setup_sys_prompt

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        load_dotenv(dotenv_path=TEST_DATA_PATH + '/.env')
        self.AI = None
        self.ClipboardAI = None

    def create_interface_objects(self, **kwargs):
        # 文档中，load_model使用class_name作为参数初始化对应类实例
        # 尝试根据class_name区分初始化类
        class_name = kwargs.pop('class_name', None)
        if class_name == 'AI' or (class_name is None):  # 默认初始化AI类
            # AI类初始化参数提取
            model_name = kwargs.pop('model_name', 'gpt-4-turbo')
            temperature = kwargs.pop('temperature', 0.1)
            azure_endpoint = kwargs.pop('azure_endpoint', None)
            streaming = kwargs.pop('streaming', True)
            vision = kwargs.pop('vision', False)
            self.AI = AI(
                model_name=model_name,
                temperature=temperature,
                azure_endpoint=azure_endpoint,
                streaming=streaming,
                vision=vision,
            )
        if class_name == 'ClipboardAI' or (class_name is None):
            self.ClipboardAI = ClipboardAI()

        # 所以如果class_name=None，默认两个类都初始化了
        self.result.fuc_name = 'create_interface_objects'
        self.result.is_success = True
        self.result.fail_reason = ''
        self.result.interface_return = None
        self.result.except_data = None

    def run(self, name: str, **kwargs):
        try:
            # 路由规则说明：
            # 函数：
            #   serialize_messages (文档的全局函数)
            # 类方法：
            #   AI.__init__无调用接口
            #   AI属性无需调用run
            #   AI方法命名规则：run('AI_方法名', **kwargs)
            #   ClipboardAI方法命名类似：run('ClipboardAI_方法名', **kwargs)

            # 处理AI类的方法
            if name == 'AI_start':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                # 参数：system (str), user (Any), step_name (str)
                system = kwargs.get('system')
                user = kwargs.get('user')
                step_name = kwargs.get('step_name')
                ret = self.AI.start(system, user, step_name=step_name)
                self.result.interface_return = ret

            elif name == 'AI_next':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                # 参数：messages(List[Message]), prompt(Optional[str]), step_name(str)
                messages = kwargs.get('messages')
                prompt = kwargs.get('prompt', None)
                step_name = kwargs.get('step_name')
                ret = self.AI.next(messages, prompt, step_name=step_name)
                self.result.interface_return = ret

            elif name == 'AI_backoff_inference':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                messages = kwargs.get('messages')
                ret = self.AI.backoff_inference(messages)
                self.result.interface_return = ret

            elif name == 'AI_serialize_messages':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                messages = kwargs.get('messages')
                ret = self.AI.serialize_messages(messages)
                self.result.interface_return = ret

            elif name == 'AI_deserialize_messages':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                jsondictstr = kwargs.get('jsondictstr')
                ret = self.AI.deserialize_messages(jsondictstr)
                self.result.interface_return = ret

            elif name == 'AI__create_chat_model':
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model with class_name='AI' first.")
                ret = self.AI._create_chat_model()
                self.result.interface_return = ret

            # 处理ClipboardAI类的方法
            elif name == 'ClipboardAI_serialize_messages':
                if self.ClipboardAI is None:
                    raise Exception("ClipboardAI instance not initialized. Please run load_model with class_name='ClipboardAI' first.")
                messages = kwargs.get('messages')
                ret = self.ClipboardAI.serialize_messages(messages)
                self.result.interface_return = ret

            elif name == 'ClipboardAI_multiline_input':
                if self.ClipboardAI is None:
                    raise Exception("ClipboardAI instance not initialized. Please run load_model with class_name='ClipboardAI' first.")
                ret = self.ClipboardAI.multiline_input()
                self.result.interface_return = ret

            elif name == 'ClipboardAI_next':
                if self.ClipboardAI is None:
                    raise Exception("ClipboardAI instance not initialized. Please run load_model with class_name='ClipboardAI' first.")
                messages = kwargs.get('messages')
                prompt = kwargs.get('prompt', None)
                step_name = kwargs.get('step_name')
                ret = self.ClipboardAI.next(messages, prompt, step_name=step_name)
                self.result.interface_return = ret

            # 处理全局函数 serialize_messages
            elif name == 'serialize_messages':
                # 全局方法 serialize_messages(messages)
                messages = kwargs.get('messages')
                if self.AI is None:
                    raise Exception("AI instance not initialized. Please run load_model first.")
                ret = self.AI.serialize_messages(messages)
                self.result.interface_return = ret

            else:
                raise Exception(f"Unknown function or method name: {name}")

            self.result.fuc_name = name
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.is_file = False
            self.result.file_path = ''
            self.result.except_data = None

        except Exception as e:
            self.result.fuc_name = name
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            self.result.is_file = False
            self.result.file_path = ''
            self.result.except_data = None

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('AI_start')
adapter_addtional_data['functions'].append('AI_next')
adapter_addtional_data['functions'].append('AI_backoff_inference')
adapter_addtional_data['functions'].append('AI_serialize_messages')
adapter_addtional_data['functions'].append('AI_deserialize_messages')
adapter_addtional_data['functions'].append('AI__create_chat_model')
adapter_addtional_data['functions'].append('ClipboardAI_serialize_messages')
adapter_addtional_data['functions'].append('ClipboardAI_multiline_input')
adapter_addtional_data['functions'].append('ClipboardAI_next')
adapter_addtional_data['functions'].append('serialize_messages')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)