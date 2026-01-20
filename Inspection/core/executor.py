import os
import sys
import importlib
import time
import inspect
import json
from pathlib import PosixPath, WindowsPath
from Inspection.utils.path_manager import (
    RECORD_PATH,
    CUSTOM_ADAPTER_PATH,
    RESOURCES_PATH,
)
from Inspection.utils.shared_config import OSENV_CONFIG as CONFIG
from Inspection.utils.tools import (
    FileGuardHandler,
    ReadOnlyGuardHandler,
    serialize_payload_to_json,
    save_pkl_with_limit,
)
from Inspection.adapters.base_adapter import BaseAdapter
from watchdog.observers import Observer


class Executor:
    if os.name == "nt":
        _BasePath = WindowsPath
    else:
        _BasePath = PosixPath

    class SmartPath(_BasePath):
        def __add__(self, other):
            return str(self) + str(other)

        def __radd__(self, other):
            return str(other) + str(self)

    def __init__(self, project_name: str, exe_type: str = "pre"):
        self.project_name = project_name
        self.adapter_path = ""
        self.date_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.result = None
        self.exe_type = exe_type
        self.time = ""
        self.current_args = None
        self.adapter_path = os.path.join(CUSTOM_ADAPTER_PATH, f"{self.project_name}.py")
        if not os.path.exists(self.adapter_path):
            raise FileNotFoundError(
                f"Adapter file {self.adapter_path} does not exist, please check if the adapter name or path is correct"
            )
        self.adapter: BaseAdapter = self.load_adapter()
        self.record_functions = []
        self.record_times = 0
        self.now_record_path = self.SmartPath("")
        self.funs_exec_times = {}
        self.observer = None  # 文件监控对象
        self.readonly_observer = None  # 只读监控对象
        self.__create_now_record_path()
        self.__setup_readonly_guard()  # 设置只读守护

    def __create_now_record_path(self):
        temp_record_path = CONFIG.get("temp_record_path", "")
        if temp_record_path != "":
            record_path = os.path.join(
                temp_record_path,
                f"{self.exe_type}",
                f"{self.project_name}_{self.date_str}",
            )
        else:
            record_path = os.path.join(
                RECORD_PATH, f"{self.exe_type}", f"{self.project_name}_{self.date_str}"
            )
        self.now_record_path = self.SmartPath(record_path + "/")
        os.makedirs(record_path, exist_ok=True)

        # 启动实时文件守护
        try:
            event_handler = FileGuardHandler(record_path, max_files=200)
            self.observer = Observer()
            self.observer.daemon = True  # 设置为守护线程，主程序退出时自动结束
            self.observer.schedule(event_handler, record_path, recursive=False)
            self.observer.start()
        except:
            pass

    def create_interface_objects(self, interface_class_name, **kwargs):
        print(f'[INS_INFO] Running "create_interface_objects" function')
        self.current_args = kwargs
        self.time = time.time()
        self.adapter.create_interface_objects(interface_class_name, **kwargs)
        if not self.adapter.result.func_name:
            self.adapter.result.func_name = "create_interface_objects"
        self.result = self.adapter.result
        if not self.result.is_success:
            print(self.result.fail_reason)
        if (
            not self.record_functions
            or "create_interface_objects" in self.record_functions
        ):
            self.record_execution_result()
        return self.adapter.result.interface_return

    def run(self, function_name, **kwargs):
        self.funs_exec_times[function_name] = (
            self.funs_exec_times.get(function_name, 0) + 1
        )
        if self.funs_exec_times[function_name] <= 100:
            print(f'[INS_INFO] Running interface "{function_name}"')
            if self.funs_exec_times[function_name] == 100:
                print(
                    f'[INS_WARN] Function "{function_name}" has been run more than 100 times, execution results will no longer be recorded'
                )
        self.current_args = kwargs
        self.time = time.time()
        self.adapter.run(function_name, **kwargs)
        self.result = self.adapter.result
        if not self.result.is_success:
            print(self.result.fail_reason)
        if self.funs_exec_times[function_name] <= 100:
            if not self.record_functions or function_name in self.record_functions:
                self.record_execution_result()
        return self.adapter.result.interface_return

    def set_record_function(self, function_names=[]):
        if not isinstance(function_names, list):
            if isinstance(function_names, str):
                function_names = [function_names]
            else:
                raise ValueError("function_names should be a list or a string")
        self.record_functions = function_names

    def load_adapter(self):
        module_name = os.path.splitext(os.path.basename(self.adapter_path))[0]
        full_module_name = f"Inspection.adapters.custom_adapters.{module_name}"
        if full_module_name in sys.modules:
            del sys.modules[full_module_name]
        module = importlib.import_module(full_module_name)
        adapter_class = getattr(module, "CustomAdapter", None)
        if adapter_class is None:
            classes = inspect.getmembers(module, inspect.isclass)
            if not classes:
                raise AttributeError(f"No class found in {self.adapter_path}")
            adapter_class = classes[0][1]
            print(
                f"[INS_WARN] Adapter class not found, using first class: {adapter_class.__name__}"
            )
        return adapter_class()

    def record_execution_result(self):
        self.record_times += 1
        # Get current result
        result = self.result
        if not result:
            raise ValueError("No execution result to record")
        sys_path = sys.path.copy()
        process_time = time.time() - self.time
        print(f"[INS_INFO] Execution time: {process_time:.2f} seconds")
        # Result save directory
        path = os.path.join(
            str(self.now_record_path), f"{self.record_times}_{result.func_name}"
        )
        os.makedirs(path, exist_ok=True)

        record_pkl = CONFIG.get("record_pkl", False)
        result_dict = {}
        # 保存 json
        try:
            result_dict = serialize_payload_to_json(result.__dict__)
            result_dict["process_time"] = process_time
            result_dict["args"] = serialize_payload_to_json(self.current_args)
            result_dict["sys_path"] = serialize_payload_to_json(sys_path)

            with open(
                os.path.join(path, "result_data.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(result_dict, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[INS_WARN] Result cannot be saved as JSON format: {e}")
            record_pkl = True  # 如果json保存失败，尝试保存为pkl

        # 保存 pkl
        if record_pkl:
            try:
                # 从配置读取最大大小（MB），默认200
                
                result_file = os.path.join(path, "result_data.pkl")
                
                other_data_dic = {
                    "sys_path": sys_path,
                    "args": self.current_args,
                    "process_time": process_time,
                }
                save_pkl_with_limit(result_file, other_data_dic, result)
                if result_dict == {}:
                    result_dict = result.__dict__
            except Exception as e:
                print(f"[INS_WARN] Result cannot be saved as binary format: {e}")
                return

        # 如果是文件，复制到目标目录
        if result_dict.get("is_file"):
            file_path = getattr(result, "file_path", "")
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    os.system(f'cp "{file_path}" "{path}"')
                    print(f"[INS_INFO] File {file_path} has been copied to {path}")
                elif os.path.isdir(result_dict["file_path"]):
                    os.system(f'cp -r "{result_dict["file_path"]}" "{path}"')
                    print(
                        f'[INS_INFO] Folder {result_dict["file_path"]} has been copied to {path}'
                    )
            else:
                print("[INS_WARN] File not found")

    def __setup_readonly_guard(self):
        """为RESOURCES_PATH设置只读守护"""
        try:
            if os.path.exists(RESOURCES_PATH):
                event_handler = ReadOnlyGuardHandler(RESOURCES_PATH)
                self.readonly_observer = Observer()
                self.readonly_observer.daemon = True  # 设置为守护线程，主程序退出时自动结束
                self.readonly_observer.schedule(
                    event_handler, RESOURCES_PATH, recursive=True
                )
                self.readonly_observer.start()
        except Exception as e:
            print(f"[INS_WARN] Failed to setup readonly guard: {e}")


if __name__ == "__main__":
    executor = Executor("example")
    executor.run("example_function", param1="value1", param2=123)
