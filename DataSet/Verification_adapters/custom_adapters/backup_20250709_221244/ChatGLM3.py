# ChatGLM3 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'ChatGLM3/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/ChatGLM3')

# 可以在此位置后添加导包部分代码
import os
import platform
from transformers import AutoTokenizer
from transformers import AutoModel


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.auto_tokenizer = None
        self.auto_model = None

    def create_interface_objects(self, **kwargs):
        model_path = kwargs.get('model_path', None)
        if model_path:
            self.auto_model = AutoModel.from_pretrained(model_path, **kwargs)
            self.auto_tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return={'model': self.auto_model, 'tokenizer': self.auto_tokenizer}
            )
        else:
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=False,
                fail_reason='model_path is required',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

    def run(self, name: str, **kwargs):
        if name == 'from_pretrained':
            try:
                model = AutoModel.from_pretrained(**kwargs)
                self.result.set_result(
                    fuc_name='from_pretrained',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=model
                )
            except Exception as e:
                self.result.set_result(
                    fuc_name='from_pretrained',
                    is_success=False,
                    fail_reason=str(e),
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
        
        elif name == 'from_config':
            try:
                config = kwargs.get('config', None)
                if config is None:
                    raise ValueError("config is required")
                model = AutoModel.from_config(config, **kwargs)
                self.result.set_result(
                    fuc_name='from_config',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=model
                )
            except Exception as e:
                self.result.set_result(
                    fuc_name='from_config',
                    is_success=False,
                    fail_reason=str(e),
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'register':
            try:
                config_class = kwargs.get('config_class', None)
                model_class = kwargs.get('model_class', None)
                if config_class is None or model_class is None:
                    raise ValueError("config_class and model_class are required")
                AutoModel.register(config_class, model_class, **kwargs)
                self.result.set_result(
                    fuc_name='register',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            except Exception as e:
                self.result.set_result(
                    fuc_name='register',
                    is_success=False,
                    fail_reason=str(e),
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'from_pretrained_tokenizer':
            try:
                tokenizer = AutoTokenizer.from_pretrained(**kwargs)
                self.result.set_result(
                    fuc_name='from_pretrained_tokenizer',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=tokenizer
                )
            except Exception as e:
                self.result.set_result(
                    fuc_name='from_pretrained_tokenizer',
                    is_success=False,
                    fail_reason=str(e),
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'register_tokenizer':
            try:
                config_class = kwargs.get('config_class', None)
                slow_tokenizer_class = kwargs.get('slow_tokenizer_class', None)
                fast_tokenizer_class = kwargs.get('fast_tokenizer_class', None)
                if config_class is None:
                    raise ValueError("config_class is required")
                AutoTokenizer.register(config_class, slow_tokenizer_class, fast_tokenizer_class, **kwargs)
                self.result.set_result(
                    fuc_name='register_tokenizer',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            except Exception as e:
                self.result.set_result(
                    fuc_name='register_tokenizer',
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
adapter_additional_data['functions'].append('from_pretrained')
adapter_additional_data['functions'].append('from_config')
adapter_additional_data['functions'].append('register')
adapter_additional_data['functions'].append('from_pretrained_tokenizer')
adapter_additional_data['functions'].append('register_tokenizer')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
