import os
from Inspection.adapters import ExecutionResult
from Inspection import RECORD_PATH
from typing import Any, Tuple, List
import sys

SYS_RAW_PATH = sys.path

class RecordPickleReader:
    def __init__(self, pkl_path: str):
        """ 初始化 PickleReader，指定 .pkl 文件的路径 """
        self.pkl_path = pkl_path
        self.other_data_dic = {}  # 用于存储加载的其他数据
        self.result: ExecutionResult = None  # 用于存储加载的执行结果对象
        try:
            self.__load()  # 调用加载方法
        except Exception as e:
            print(f"Failed to load pickle file {self.pkl_path}: {e}")

    def __load(self):
        """ 从指定的 .pkl 文件加载数据 
            返回元组的第一个元素为其他数据字典，第二个为执行结果对象
        """
        with open(self.pkl_path, 'rb') as f:
            import dill
            other_data_dic = dill.load(f)
            # 将不同的sys.path添加到当前环境
            for path in other_data_dic['sys_path']:
                if path not in sys.path:
                    sys.path.append(path)
                    # print(f"Adding path: {path}")
            result = dill.load(f)
            self.other_data_dic = other_data_dic
            self.result = result
    
    def get_args(self) -> Any:
        """ 获取 .pkl 文件中的 args """
        if self.result is None:
            return None
        return self.other_data_dic.get('args', None)
    
    def get_interface_return(self) -> Any:
        """ 获取 .pkl 文件中的 interface_return """
        if self.result is None:
            return None
        return self.result.interface_return

    def get_except_data(self) -> Any:
        """ 获取 .pkl 文件中的 except_data """
        if self.result is None:
            return None
        try:
            result = self.result.except_data
        except AttributeError:
            # 兼容旧版本的 except_data 字段
            result = self.result.raw_data if hasattr(self.result, 'raw_data') else None
        return result
    
    def get_is_success(self) -> bool:
        """ 获取 .pkl 文件中的 is_success """
        if self.result is None:
            return False
        return self.result.is_success
    
    def get_fail_reason(self) -> str:
        """ 获取 .pkl 文件中的 fail_reason """
        if self.result is None:
            return None
        return self.result.fail_reason

    def print_data(self):
        """ 打印 .pkl 文件中的数据 """
        args = self.other_data_dic.get('args', None)
        process_time = self.other_data_dic.get('process_time', None)
        result = self.result
        if result is not None and args is not None:
            print(f"  Function name: {result.fuc_name}")
            print(f"  Processing time: {process_time:.2f} seconds")
            print(f"  Success: {result.is_success}")
            if not result.is_success:
                print(f"  Failure reason: {result.fail_reason}")
            print(f"  Is file: {result.is_file}")
            if result.is_file:
                print(f"  File path: {result.file_path}")
            print(f"  Expected data: {result.except_data}")
            print(f"  Interface return: {result.interface_return}")
            print(f"  Input parameters: {args}")
            if result.is_file:
                print("File has been copied to current execution function result directory and can be viewed directly")
            print("-" * 40)


class RecordJsonReader:
    def __init__(self, json_path: str = ''):
        """ 初始化 JsonReader，指定 .json 文件的路径 """
        self.json_path = json_path
        self.data_dic = {}  # 用于存储加载的 JSON 数据
        self.__load()  # 调用加载方法

    def __load(self) -> Tuple[Any, Any]:
        """ 从指定的 .json 文件加载数据 """
        data_dic = {}
        import json
        try:
            with open(self.json_path, 'r') as f:
                data_dic = json.load(f)
        except Exception as e:
            print(f"Failed to load json file {self.json_path}: {e}")
        self.data_dic = data_dic

    def get_args(self) -> Any:
        """ 获取 JSON 数据中的 args """
        return self.data_dic.get('args', None)
        
    def get_interface_return(self) -> Any:
        """ 获取 JSON 数据中的 interface_return """
        return self.data_dic.get('interface_return', None)
    
    def get_except_data(self) -> Any:
        """ 获取 JSON 数据中的 except_data """
        result = self.data_dic.get('except_data', None)
        if result is None:
            result = self.data_dic.get('raw_data', None) #为了兼容旧版本
        return result
    
    def get_is_success(self) -> bool:
        """ 获取 JSON 数据中的 is_success """
        return self.data_dic.get('is_success', False)
    
    def get_fail_reason(self) -> str:
        """ 获取 JSON 数据中的 fail_reason """
        return self.data_dic.get('fail_reason', '')
    
    def print_data(self):
        """ 打印 JSON 数据 """
        args = self.data_dic.get('args', None)
        process_time = self.data_dic.get('process_time', None)
        result = self.data_dic
        if result is not None:
            print(f"  Function name: {result['fuc_name']}")
            print(f"  Processing time: {process_time:.2f} seconds")
            print(f"  Success: {result['is_success']}")
            if not result['is_success']:
                print(f"  Failure reason: {result['fail_reason']}")
            print(f"  Is file: {result['is_file']}")
            if result['is_file']:
                print(f"  File path: {result['file_path']}")
            print(f"  Expected data: {result['except_data']}")
            print(f"  Interface return: {result['interface_return']}")
            print(f"  Input parameters: {args}")
            if result['is_file']:
                print("File has been copied to current execution function result directory and can be viewed directly")
            print("-" * 40)


class ExecutionResultReader:
    def __init__(self, base_dir: str):
        """ 初始化读取器，指定Records文件夹的根目录 """
        self.base_dir = base_dir  # Records文件夹的根目录
        self.dill_reader: RecordPickleReader = None  # PickleReader对象
        self.json_reader: RecordJsonReader = None  # JsonReader对象

    def list_projects(self,path=None) -> List[str]:
        """ 列出所有项目文件夹 """
        if path is None:
            return [name for name in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, name))]
        else:
            return [name for name in os.listdir(self.base_dir+'/'+path) if os.path.isdir(os.path.join(self.base_dir+'/'+path, name))]

    def list_function_folders(self, project_dir: str) -> List[str]:
        """ 列出某个项目文件夹下的所有函数文件夹 """
        project_path = os.path.join(self.base_dir, project_dir)
        return [f for f in os.listdir(project_path) if os.path.isdir(os.path.join(project_path, f))]

    def load_results_from_folder(self, project_dir: str, func_folder: str) -> None:
        """ 从指定项目的指定函数文件夹加载ExecutionResult对象 """
        folder_path = os.path.join(self.base_dir, project_dir, func_folder)
        # 检查文件夹中是否存在.pkl或.json文件
        pkl_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        try:
            pkl_file = pkl_files[0]
            file_path = os.path.join(folder_path, pkl_file)
            self.dill_reader = RecordPickleReader(file_path)
            self.json_reader = None
        except Exception as e:
            print("[INS_WARN] Failed to load .pkl file:", e, "Trying to load .json file")
            if json_files:
                json_file = json_files[0]
                file_path = os.path.join(folder_path, json_file)
                self.json_reader = RecordJsonReader(file_path)
                self.dill_reader = None
            else:
                print(f"[INS_WARN] No .pkl or .json files found in this folder")
                self.dill_reader = None
                self.json_reader = None


    def display_results(self) -> None:
        if self.dill_reader is not None:
            self.dill_reader.print_data()
        elif self.json_reader is not None:
            self.json_reader.print_data()
        else:
            print("No data loaded.")
            return

class RecordCompressor:
    def __init__(self, record_path: str , output_path: str):
        """ 初始化记录压缩器，指定记录文件夹和输出路径 """
        # record_path: 执行记录文件夹，通常以dumb或simulation结尾，内部为标准exe.run的执行记录结果
        self.record_path = record_path
        self.output_path = output_path + '/compressed_records'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path , exist_ok=True)

    def compress_from_records(self):
        from Inspection.utils.file_lister import FileLister
        going_packed_pjs = FileLister(self.record_path, 'dir')._file_list
        for pj in going_packed_pjs:
            record_dirs_path = os.path.join(self.record_path, pj)
            api_dirs = FileLister(record_dirs_path, 'dir')._file_list
            for api_dir in api_dirs:
                api_dir_path = os.path.join(record_dirs_path, api_dir)
                # 检查是否有.pkl或.json文件
                json_file = [f for f in os.listdir(api_dir_path) if f == 'result_data.json']
                if len(json_file) != 1:
                    continue
                json_path = os.path.join(api_dir_path, json_file[0])
                print(f"Compressing record files for project {pj} API {api_dir}")
                compressed_json_data = self.compress_json(json_path)
                # 构造一样的目录结构
                output_dir = os.path.join(self.output_path, pj, api_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                output_file_path = os.path.join(output_dir, 'result_data.json')
                with open(output_file_path, 'w') as f:
                    import json
                    json.dump(compressed_json_data, f, indent=4)

    def compress_json(self, json_path: str):
        """
        压缩JSON文件中的特定字段
        - args字段中的每个字典值转为str(value)，长度不超过2000
        - except_data和interface_return字段的值同样处理
        """
        import json
        
        def truncate_string(value, max_length=3000):
            """将值转换为字符串并截断到指定长度"""
            str_value = str(value)
            if len(str_value) > max_length:
                return str_value[:max_length] + "..."
            return str_value
        
        def compress_dict_values(data_dict):
            """压缩字典中的值"""
            if not isinstance(data_dict, dict):
                return truncate_string(data_dict)
            
            compressed_dict = {}
            for key, value in data_dict.items():
                compressed_dict[key] = truncate_string(value)
            return compressed_dict
        # 读取JSON文件
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to read JSON file: {e}")
            return None
        # 压缩args字段
        if 'args' in data and data['args'] is not None:
            if isinstance(data['args'], dict):
                data['args'] = compress_dict_values(data['args'])
            else:
                data['args'] = truncate_string(data['args'])
        # 压缩except_data字段
        if 'except_data' in data and data['except_data'] is not None:
            if isinstance(data['except_data'], dict):
                data['except_data'] = compress_dict_values(data['except_data'])
            else:
                data['except_data'] = truncate_string(data['except_data'])
        # 压缩interface_return字段
        if 'interface_return' in data and data['interface_return'] is not None:
            if isinstance(data['interface_return'], dict):
                data['interface_return'] = compress_dict_values(data['interface_return'])
            else:
                data['interface_return'] = truncate_string(data['interface_return'])
        return data
        










# 用户交互式选择项目和记录文件夹的过程
def explore_project(reader: ExecutionResultReader):
    # 列出所有项目
    projects = reader.list_projects()
    if not projects:
        print("No projects found.")
        return None

    print("Select the execution result category to view:")
    for idx, project in enumerate(projects):
        print(f"{idx + 1}. {project}")
    try:
        project_choice = input("Please enter the project number you want to explore, or enter 'quit' to exit: ")
        if project_choice.lower() == 'quit' or project_choice.lower() == 'q':
            print('Bye！')
            return 'back'
        project_choice = int(project_choice) - 1
        if project_choice < 0 or project_choice >= len(projects):
            print("Invalid project selection.")
            return None
        project_dir = projects[project_choice]
        projectlist = reader.list_projects(project_dir)
        if not projectlist:
            print(f"\nNo projects found under {project_dir}.")
            return None
        print(f"\nAvailable projects under {project_dir}:")
        for idx, folder in enumerate(projectlist):
            print(f"{idx + 1}. {folder}")
        folder_choice = input("Please enter the project number you want to explore")
        folder_choice = int(folder_choice) - 1
        if folder_choice < 0 or folder_choice >= len(projectlist):
            print("Invalid project selection.")
            return None
        project_dir = project_dir + '/' + projectlist[folder_choice]

        result = explore_function_folder(reader, project_dir)
        if result == 'quit':
            print('Bye！')
            return 'back'
    except ValueError:
        print("Invalid input, please enter a valid number.")


def explore_function_folder(reader: ExecutionResultReader, project_dir: str):
    # 列出该项目下的所有函数文件夹
    function_folders = reader.list_function_folders(project_dir)
    if not function_folders:
        print(f"\nNo function folders found under project {project_dir}.")
        return

    print(f"\nAvailable function folders under {project_dir} project:")
    for idx, folder in enumerate(function_folders):
        print(f"{idx + 1}. {folder}")

    try:
        folder_choice = input("Please enter the function folder number you want to explore, or enter 'back' to return, or 'quit' to exit: ")
        if folder_choice.lower() == 'back' or folder_choice.lower() == 'b':
            return 'back'
        if folder_choice.lower() == 'quit' or folder_choice.lower() == 'q':
            return 'quit'
        folder_choice = int(folder_choice) - 1
        if folder_choice < 0 or folder_choice >= len(function_folders):
            print("Invalid function folder selection.")
            return
        func_folder = function_folders[folder_choice]

        # 加载并显示该函数文件夹中的ExecutionResult
        reader.load_results_from_folder(project_dir, func_folder)
        reader.display_results()

        # 用户选择查看完记录后返回上一层
        while True:
            back_or_continue = input("Please enter 'back'(or 'b') to return to the previous level, or enter a number to continue exploring the corresponding function: ")
            if back_or_continue.lower() == 'back' or back_or_continue.lower() == 'b':
                sys.path = SYS_RAW_PATH
                return 'back'
            try:
                folder_choice = int(back_or_continue) - 1
                if folder_choice < 0 or folder_choice >= len(function_folders):
                    print("Invalid number, please try again")
                    continue
                # 继续加载并显示新的函数文件夹的ExecutionResult
                func_folder = function_folders[folder_choice]
                reader.load_results_from_folder(project_dir, func_folder)
                reader.display_results()
            except ValueError:
                print("Invalid input, please enter valid numbers.")

    except ValueError:
        print("Invalid input, please enter a valid number.")


def main():
    base_dir = RECORD_PATH
    reader = ExecutionResultReader(base_dir)
    while True:
        # 启动项目探索
        result = explore_project(reader)
        if result == 'back':
            break  # 返回上一层，结束程序
