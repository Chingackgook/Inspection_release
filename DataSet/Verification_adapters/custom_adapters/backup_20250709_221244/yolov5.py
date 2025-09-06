# yolov5 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'yolov5/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/yolov5')

# 可以在此位置后添加导包部分代码
from models.common import DetectMultiBackend


# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.detect_multi_backend = None  # DetectMultiBackend 类的对象

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'DetectMultiBackend':
                # 创建接口对象
                self.detect_multi_backend = DetectMultiBackend(**kwargs)
                self.result.interface_return = self.detect_multi_backend
            elif interface_class_name == '':
                # 如果缺省，创建默认接口对象
                self.detect_multi_backend = DetectMultiBackend(**kwargs)
                self.result.interface_return = self.detect_multi_backend
            
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
            if name == 'forward':
                # 执行 forward 方法
                self.result.interface_return = self.detect_multi_backend.forward(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = name
            elif name == 'warmup':
                # 执行 warmup 方法
                self.result.interface_return = self.detect_multi_backend.warmup(**kwargs)
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
adapter_additional_data['functions'].append('DetectMultiBackend')
adapter_additional_data['functions'].append('')
adapter_additional_data['functions'].append('forward')
adapter_additional_data['functions'].append('warmup')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)


