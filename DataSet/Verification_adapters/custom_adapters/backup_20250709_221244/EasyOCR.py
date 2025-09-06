# EasyOCR 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'EasyOCR/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('')

# 可以在此位置后添加导包部分代码
import easyocr


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.reader_obj = None  # Reader类的接口对象
        try:
            if interface_class_name == 'Reader':
                # 创建Reader接口对象
                self.reader_obj = easyocr.Reader(**kwargs)
                self.result.interface_return = self.reader_obj
            elif interface_class_name == '':
                # 如果缺省，创建默认Reader接口对象
                self.reader_obj = easyocr.Reader(**kwargs)
                self.result.interface_return = self.reader_obj
            
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
            if name == 'detect':
                # 执行Reader的detect方法
                self.result.interface_return = self.reader_obj.detect(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'recognize':
                # 执行Reader的recognize方法
                self.result.interface_return = self.reader_obj.recognize(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'readtext':
                # 执行Reader的readtext方法
                self.result.interface_return = self.reader_obj.readtext(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'readtextlang':
                # 执行Reader的readtextlang方法
                self.result.interface_return = self.reader_obj.readtextlang(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'readtext_batched':
                # 执行Reader的readtext_batched方法
                self.result.interface_return = self.reader_obj.readtext_batched(**kwargs)
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
adapter_additional_data['functions'].append('Reader')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('detect')
adapter_additional_data['functions'].append('recognize')
adapter_additional_data['functions'].append('readtext')
adapter_additional_data['functions'].append('readtextlang')
adapter_additional_data['functions'].append('readtext_batched')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)


