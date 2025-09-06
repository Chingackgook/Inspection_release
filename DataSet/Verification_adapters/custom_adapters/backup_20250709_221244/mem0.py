# mem0 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'mem0/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/mem0')
# 以上是自动生成的代码，请勿修改



from typing import Any, Dict
from mem0 import Memory  # Replace with the actual module name where Memory is defined

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.memory_instance = None

    def create_interface_objects(self, config: Dict[str, Any] = None):
        """
        Initialize the Memory instance with the provided configuration.
        """
        try:
            self.memory_instance = Memory(config=config)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.memory_instance
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
        Execute the specified method of the Memory class based on the name parameter.
        """
        try:
            if name == 'add':
                result = self.memory_instance.add(**kwargs)
            elif name == 'get':
                result = self.memory_instance.get(**kwargs)
            elif name == 'get_all':
                result = self.memory_instance.get_all(**kwargs)
            elif name == 'search':
                result = self.memory_instance.search(**kwargs)
            elif name == 'update':
                result = self.memory_instance.update(**kwargs)
            elif name == 'delete':
                result = self.memory_instance.delete(**kwargs)
            elif name == 'delete_all':
                result = self.memory_instance.delete_all(**kwargs)
            elif name == 'history':
                result = self.memory_instance.history(**kwargs)
            elif name == 'reset':
                self.memory_instance.reset()
                result = {'message': 'Memory store reset successfully.'}
            elif name == 'from_config':
                result = Memory.from_config(**kwargs)
                self.memory_instance = result
            else:
                raise ValueError(f"Method '{name}' is not recognized.")

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

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('add')
adapter_addtional_data['functions'].append('get')
adapter_addtional_data['functions'].append('get_all')
adapter_addtional_data['functions'].append('search')
adapter_addtional_data['functions'].append('update')
adapter_addtional_data['functions'].append('delete')
adapter_addtional_data['functions'].append('delete_all')
adapter_addtional_data['functions'].append('history')
adapter_addtional_data['functions'].append('reset')
adapter_addtional_data['functions'].append('from_config')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
