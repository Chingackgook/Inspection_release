from typing import Any
from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import extract_python_code
from Inspection.utils.interface_reader import InterfaceDocReader , InterfaceInfoReader
from Inspection import INTERFACE_DOC_PATH, BASE_DIR
from Inspection.utils.config import CONFIG
import os
import re
from Inspection.utils.path_manager import ENV_BASE

GEN_PATH = BASE_DIR + '/Inspection/adapters/custom_adapters/'

def get_import_star_statement(project_root: str, module_path: str) -> str:
    """
    给定项目根目录和模块的完整路径，返回 `from xxx.yyy import *` 形式的导入语句。
    
    参数:
    - project_root: Python 项目的根目录
    - module_path: 要导入的模块的完整路径（.py 文件）
    
    返回:
    - from xxx.yyy import * 形式的字符串
    """
    project_root = os.path.abspath(project_root)
    module_path = os.path.abspath(module_path)
    # 去掉根目录前缀和 .py 后缀
    try:
        rel_path = os.path.relpath(module_path, project_root)
    except ValueError:
        print(f"[INS_ERR] Module path {module_path} is not under project root {project_root}")
        return ""

    if os.path.basename(rel_path) == '__init__.py':
        rel_path = os.path.dirname(rel_path)
    elif rel_path.endswith('.py'):
        rel_path = rel_path[:-3]
    elif rel_path.endswith('.pyc'):
        rel_path = rel_path[:-4]
    elif rel_path.endswith('.ipynb'):
        rel_path = rel_path[:-6]

    # 转换路径为模块导入路径
    import_path = rel_path.replace(os.path.sep, '.')
    if import_path.startswith('.'):
        import_path = import_path[1:]  # 去掉开头的点
    if not import_path:
        return ""
    return f"from {import_path} import *"


class AdapterGenerator:
    def __init__(self, name):
        self.name = name
        self.base_ai = Any
        self.ask = CONFIG.get('ask', True)
        self.analysis_temprature = CONFIG.get('adapter_analysis_temprature', 0.5)
        self.generate_code_temprature = CONFIG.get('adapter_generate_code_temprature', 0.3)

    def set_base_ai(self, base_ai: BaseAI):
        self.base_ai = base_ai

    def analyze_project_root_path(self, path):
        abs_path = path
        project_name = self.name
        # 只保留字母和数字进行模糊匹配
        def normalize(s):
            return re.sub(r'[^a-zA-Z0-9]', '', s).lower()
        norm_project = normalize(project_name)
        parts = abs_path.split(os.sep)
        matched_indices = [
            idx for idx, part in enumerate(parts) if norm_project in normalize(part)
        ]
        if matched_indices:
            index = matched_indices[0]
            return os.sep.join(parts[:index + 1])
        else:
            return abs_path
    
    def generate_adapter(self):
        try:
            doc = InterfaceDocReader(self.name).get_doc()
            with open(BASE_DIR + '/Inspection/adapters/base_adapter.py', 'r') as f:
                base_adapter = f.read()
        except Exception:
            print(f'[INS_ERR] Adapter generation failed, please check if interface documentation exists in {INTERFACE_DOC_PATH}')
            return
        
        if os.path.exists(GEN_PATH + self.name + '.py') and self.ask:
            print(f"[INS_WARN] Adapter {self.name} already exists")
            ch = input("Overwrite? (y/n)")
            if ch != 'y':
                return
        way_save_path = GEN_PATH + 'way/'
        if not os.path.exists(way_save_path):
            os.makedirs(way_save_path)
        
        info_reader = InterfaceInfoReader(self.name)
        project_root = info_reader.get_project_root()
        implement_data = info_reader.get_implementation_list()

        import_interface_statements = ""
        for data in implement_data:
            code_path = data.get('Path', '')
            if code_path:
                import_interface_statement = get_import_star_statement(project_root, code_path)
                if import_interface_statement and import_interface_statement not in import_interface_statements:
                    import_interface_statements += import_interface_statement + "\n"
                
        
        gen_way_prompt = f"""
{doc}
Here is my interface documentation, which includes several interface classes and top-level functions.
Please help me clearly classify them:
Identify which are top-level functions.
Identify which are methods, and specify the class each method belongs to, is it a static method, instance method, or class method.
Additionally, tell me the total number of interface classes.
""" 
        way_result = self.base_ai.generate_text(gen_way_prompt) + "\n\n"
        gen_way_prompt = f"""
Given the following template:
{base_adapter}\n
Please tell me how this template should be filled in. (Only explain how to fill it in; do not generate code.)
Please answer the following questions one by one:

Q1: Which interface class objects need to be initialized in create_interface_objects, or is initialization unnecessary ?(for top-level functions, initialization is not needed)
Q2: Which top-level functions should be mapped to `run`?
Q3: Which instance methods , class methods, or static methods should be mapped to `run`?(Do not omit methods that the documentation mentioned)

Please insure:
For the functions mentioned in the interface documentation, they should be mapped to the form run(function_name, **kwargs).
For the classes mentioned in the interface documentation, their methods should be mapped to the form run(class_name_method_name, **kwargs). If there is only one interface class, methods can also be mapped as run(method_name, **kwargs) directly.
"""
        way_result += self.base_ai.generate_text(gen_way_prompt)
        with open(way_save_path + self.name + '.md', 'w') as f:
            f.write(way_result)
        print(f"[INS_INFO] Adapter {self.name} generation plan saved to {way_save_path + self.name + '.md'}")
        prompt = f"""
Task:
Generate the implementation of the CustomAdapter class. Only provide the implementation of the CustomAdapter class.
You do not need to handle import statements. You can directly use the classes and functions from the interface documentation.
Please output a complete code block wrapped in:
```python  ```
"""

        result = self.base_ai.generate_text(prompt)
        result = extract_python_code(result)

        dead_code_front = f"# {self.name} \n"
        dead_code_front += "from Inspection import ENV_BASE\n"
        dead_code_front += f"ENV_DIR = ENV_BASE + '{self.name}/'\n"
        dead_code_front += "from Inspection.adapters import BaseAdapter\n"
        dead_code_front += "from Inspection.adapters import ExecutionResult\n"
        dead_code_front += "import sys\n"
        dead_code_front += "import os\n"
        dead_code_front += f"sys.path.insert(0, '{project_root}')\n"
        dead_code_front += f"os.chdir('{project_root}')\n"
        dead_code_front += f"""
# you can add your custom imports here
{import_interface_statements}
# DeadCodeFront end\n
"""
        dead_code_end = f"""
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
"""
        result_code = dead_code_front + result + dead_code_end
        with open(GEN_PATH + self.name + '.py', 'w') as f:
            f.write(result_code)
        self.__create_env_dir()
        print(f"[INS_INFO] Adapter {self.name} generated successfully, saved to {GEN_PATH + self.name + '.py'}")

    def __create_env_dir(self):
        env_dir = ENV_BASE + self.name + '/'
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)

