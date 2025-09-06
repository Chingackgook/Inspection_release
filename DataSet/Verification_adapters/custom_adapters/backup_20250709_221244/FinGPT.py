# FinGPT 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'FinGPT/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os

# 可以在此位置后添加导包部分代码


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # LlamaTokenizerFast的接口对象
        self.class2_obj = None  # LlamaForCausalLM的接口对象
        self.class3_obj = None  # PeftModel的接口对象
        
        try:
            if interface_class_name == 'LlamaTokenizerFast':
                self.class1_obj = LlamaTokenizerFast.from_pretrained(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == 'LlamaForCausalLM':
                self.class2_obj = LlamaForCausalLM.from_pretrained(**kwargs)
                self.result.interface_return = self.class2_obj
            elif interface_class_name == 'PeftModel':
                base_model = kwargs.get('base_model')
                self.class3_obj = PeftModel.from_pretrained(base_model, **kwargs)
                self.result.interface_return = self.class3_obj
            else:
                # 如果缺省，创建默认接口对象（假设只有LlamaTokenizerFast）
                self.class1_obj = LlamaTokenizerFast.from_pretrained(**kwargs)
                self.result.interface_return = self.class1_obj
            
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 创建接口对象失败: {e}")

    def run(self, name: str, **kwargs):
        try:
            if name == 'load_local_model':
                self.result.interface_return = load_local_model(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'generate_answer':
                self.result.interface_return = generate_answer(self.class2_obj, self.class1_obj, **kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'run_demo_samples':
                self.result.interface_return = run_demo_samples(self.class2_obj, self.class1_obj)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'LlamaTokenizerFast___call__':
                self.result.interface_return = self.class1_obj(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'LlamaForCausalLM_generate':
                self.result.interface_return = self.class2_obj.generate(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'PeftModel_eval':
                self.class3_obj.eval()
                self.result.interface_return = None
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            else:
                raise ValueError(f"Unknown method name: {name}")

        except Exception as e:
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] 执行方法 {name} 失败: {e}")

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('LlamaTokenizerFast')
adapter_additional_data['functions'].append('LlamaForCausalLM')
adapter_additional_data['functions'].append('PeftModel')
adapter_additional_data['functions'].append('load_local_model')
adapter_additional_data['functions'].append('generate_answer')
adapter_additional_data['functions'].append('run_demo_samples')
adapter_additional_data['functions'].append('LlamaTokenizerFast___call__')
adapter_additional_data['functions'].append('LlamaForCausalLM_generate')
adapter_additional_data['functions'].append('PeftModel_eval')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
