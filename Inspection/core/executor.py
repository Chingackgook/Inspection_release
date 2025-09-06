import os
import sys
import importlib
import time
import inspect
import json
import numpy as np
from pathlib import  PosixPath, WindowsPath
from Inspection.utils.path_manager import RECORD_PATH,CUSTOM_ADAPTER_PATH
from Inspection.utils.shared_config import OSENV_CONFIG as CONFIG
from Inspection.adapters.base_adapter import BaseAdapter


class Executor:
    if os.name == 'nt':
        _BasePath = WindowsPath
    else:
        _BasePath = PosixPath
    class SmartPath(_BasePath):
        def __add__(self, other):
            return str(self) + str(other)

        def __radd__(self, other):
            return str(other) + str(self)
    
    def __init__(self, adapter_name: str, exe_type: str = 'pre'):
        self.adapter_name = adapter_name
        self.adapter_path = ''
        self.date_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
        self.result = None
        self.exe_type = exe_type
        self.time = ''
        self.current_args = None
        self.adapter_path = os.path.join(CUSTOM_ADAPTER_PATH, f'{self.adapter_name}.py')
        if not os.path.exists(self.adapter_path):
            raise FileNotFoundError(f'Adapter file {self.adapter_path} does not exist, please check if the adapter name or path is correct')
        self.adapter :BaseAdapter = self.load_adapter()  # instantiate
        self.record_functions = []
        self.record_times = 0
        self.now_record_path = self.SmartPath('')
        self.funs_exec_times = {} #key: function name that has been run, value: number of runs
        self.__create_now_record_path()

    def set_now_record_path(self, path):
        self.now_record_path = self.SmartPath(path)

    def __create_now_record_path(self):
        temp_record_path = CONFIG.get('temp_record_path', '')
        if temp_record_path != '':
            record_path = os.path.join(temp_record_path, f'{self.exe_type}', f'{self.adapter_name}_{self.date_str}')
        else:
            record_path = os.path.join(RECORD_PATH, f'{self.exe_type}', f'{self.adapter_name}_{self.date_str}')
        self.now_record_path = self.SmartPath(record_path + '/')
        os.makedirs(record_path, exist_ok=True)

    def set_record_function(self, function_names=[]):
        if not isinstance(function_names, list):
            raise TypeError('function_names must be a list')
        self.record_functions = function_names

    def load_adapter(self):
        module_name = os.path.splitext(os.path.basename(self.adapter_path))[0]
        full_module_name = f'Inspection.adapters.custom_adapters.{module_name}'
        # If module is already in sys.modules, delete it first
        if full_module_name in sys.modules:
            del sys.modules[full_module_name]
        module = importlib.import_module(full_module_name)
        adapter_class = getattr(module, 'CustomAdapter', None)
        if adapter_class is None:
            classes = inspect.getmembers(module, inspect.isclass)
            if not classes:
                raise AttributeError(f"No class found in {self.adapter_path}")
            adapter_class = classes[0][1]
            print(f'[INS_WARN] Adapter class not found, using first class: {adapter_class.__name__}')
        return adapter_class()

    def record_execution_result(self):
        self.record_times += 1
        # Get current result
        result = self.result
        if not result:
            raise ValueError('No execution result to record')
        sys_path = sys.path.copy()
        process_time = time.time() - self.time
        print(f'[INS_INFO] Execution time: {process_time:.2f} seconds')
        # Result save directory
        path = os.path.join(str(self.now_record_path), f'{self.record_times}_{result.fuc_name}')
        os.makedirs(path, exist_ok=True)

        record_pkl = CONFIG.get('record_pkl', False)
        # 保存 json
        try:
            result_dict = result.__dict__
            result_dict['process_time'] = process_time
            # 改为保存当前参数
            result_dict['args'] = self.current_args
            result_dict['sys_path'] = sys_path
            def handle_non_serializable(obj):
                if isinstance(obj, (bytes, bytearray)):
                    return obj.decode('utf-8', errors='ignore')
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return str(obj)

            with open(os.path.join(path, 'result_data.json'), 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, default=handle_non_serializable, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f'[INS_WARN] Result cannot be saved as JSON format: {e}')
            record_pkl = True  # 如果json保存失败，尝试保存为pkl

        # 保存 pkl
        if record_pkl:
            import dill
            try:
                result_file = os.path.join(path, 'result_data.pkl')
                other_data_dic = {
                    'sys_path': sys_path,
                    'args': self.current_args,
                    'process_time': process_time,
                }
                with open(result_file, 'wb') as f:
                    dill.dump(other_data_dic, f)
                    dill.dump(result, f)
                result_dict = result.__dict__
            except Exception as e:
                print(f'[INS_WARN] Result cannot be saved as binary format: {e}')
                return

        # 如果是文件，复制到目标目录
        if result_dict.get('is_file'):
            file_path = getattr(result, 'file_path', '')
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.system(f'cp "{file_path}" "{path}"')
                    print(f'[INS_INFO] File {file_path} has been copied to {path}')
                elif os.path.isdir(result_dict['file_path']):
                    os.system(f'cp -r "{result_dict["file_path"]}" "{path}"')
                    print(f'[INS_INFO] Folder {result_dict["file_path"]} has been copied to {path}')
            else:
                print("[INS_WARN] File not found")

    def create_interface_objects(self, **kwargs):
        print(f'[INS_INFO] Running "create_interface_objects" function')
        self.current_args = kwargs
        self.time = time.time()
        self.adapter.create_interface_objects(**kwargs)
        if not self.adapter.result.fuc_name:
            self.adapter.result.fuc_name = 'create_interface_objects'
        self.result = self.adapter.result
        if not self.record_functions or 'create_interface_objects' in self.record_functions:
            self.record_execution_result()
        return self.adapter.result.interface_return

    def run(self, function_name, **kwargs):
        self.funs_exec_times[function_name] = self.funs_exec_times.get(function_name, 0) + 1
        if self.funs_exec_times[function_name] <= 100:
            print(f'[INS_INFO] Running interface "{function_name}"')
            if self.funs_exec_times[function_name] == 100:
                print(f'[INS_WARN] Function "{function_name}" has been run more than 100 times, execution results will no longer be recorded')
        self.current_args = kwargs
        self.time = time.time()
        self.adapter.run(function_name, **kwargs)
        self.result = self.adapter.result
        if self.funs_exec_times[function_name] <= 100:
            if not self.record_functions or function_name in self.record_functions:
                self.record_execution_result()
        return self.adapter.result.interface_return