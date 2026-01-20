from typing import Any, List, Dict
from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import (
    get_functions_and_class_methods,
    get_inherit_info,
    has_syntax_error,
)
from Inspection.utils.readers import InterfaceInfoReader, ProjectRegistrationInfoReader
from Inspection.utils.path_manager import BASE_DIR, ENV_BASE, INTERFACE_INFO_PATH
from Inspection.utils.config import CONFIG
from Inspection.utils.tools import to_valid_module_name
import os
import re
import ast

GEN_PATH = BASE_DIR + "/Inspection/adapters/custom_adapters/"


def get_import_star_statement(
    project_root: str, module_path: str, symbol_name: str | None = None
) -> str:
    """
    给定项目根目录和模块的完整路径，返回导入语句。

    参数:
    - project_root: Python 项目的根目录
    - module_path: 要导入的模块的完整路径（.py 文件）
    - symbol_name: 要导入的符号名（函数/类/变量）。
        - None: 返回 `from xxx.yyy import *`
        - 非 None: 返回 `from xxx.yyy import symbol_name`

    返回:
    - 对应的 import 语句字符串
    """
    project_root = os.path.abspath(project_root)
    module_path = os.path.abspath(module_path)
    # 去掉根目录前缀和 .py 后缀
    try:
        rel_path = os.path.relpath(module_path, project_root)
    except ValueError:
        print(
            f"[INS_ERR] Module path {module_path} is not under project root {project_root}"
        )
        return ""

    if os.path.basename(rel_path) == "__init__.py":
        rel_path = os.path.dirname(rel_path)
    elif rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    elif rel_path.endswith(".pyc"):
        rel_path = rel_path[:-4]
    elif rel_path.endswith(".ipynb"):
        rel_path = rel_path[:-6]

    # 转换路径为模块导入路径
    import_path = rel_path.replace(os.path.sep, ".")
    if import_path.startswith("."):
        import_path = import_path[1:]  # 去掉开头的点
    if not import_path:
        return ""

    if symbol_name is None:
        return f"from {import_path} import *"
    return f"from {import_path} import {symbol_name}"


def add_python_indent(indent_level: int, text: str) -> str:
    """
    为字符串每一行添加标准 Python 缩进（每层4个空格）。

    参数:
    - indent_level: 缩进层数（int）
    - text: 原始字符串（str）

    返回:
    - 添加缩进后的字符串（str）
    """
    indent = " " * 4 * indent_level
    return "\n".join(
        [indent + line if line.strip() else line for line in text.split("\n")]
    )


def get_method_types(code: str, class_name: str) -> Dict[str, str]:
    """
    获取指定类中每个方法的类型（staticmethod, classmethod, 或 instancemethod）

    参数:
    - code: Python 代码字符串
    - class_name: 类名

    返回:
    - Dict[method_name, method_type]，其中 method_type 可以是 'static', 'class'，或 'instance'
    """
    method_types = {}
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_name = item.name
                        # 检查装饰器
                        is_static = False
                        is_classmethod = False
                        for decorator in item.decorator_list:
                            if isinstance(decorator, ast.Name):
                                if decorator.id == "staticmethod":
                                    is_static = True
                                elif decorator.id == "classmethod":
                                    is_classmethod = True

                        if is_static:
                            method_types[method_name] = "static"
                        elif is_classmethod:
                            method_types[method_name] = "class"
                        else:
                            method_types[method_name] = "instance"
    except SyntaxError:
        print(
            f"[INS_WARN] Failed to parse code for class {class_name}, assuming all methods are instance methods"
        )

    return method_types


class AdapterGenerator:
    def __init__(self, name):
        self.name = name
        self.base_ai = Any
        self.ask = CONFIG.get("ask", True)
        self.analysis_temprature = CONFIG.get("adapter_analysis_temprature", 0.5)
        self.generate_code_temprature = CONFIG.get(
            "adapter_generate_code_temprature", 0.3
        )
        self.inherit_info = []

    def generate_adapter(self):
        if not ProjectRegistrationInfoReader().can_generate_adapter(self.name):
            print(
                f"[INS_ERROR] Project {self.name} can not generate adapter, not found {self.name} in {INTERFACE_INFO_PATH}"
            )
            return

        if os.path.exists(GEN_PATH + self.name + ".py") and self.ask:
            print(f"[INS_WARN] Adapter {self.name} already exists")
            ch = input("Overwrite? (y/n)")
            if ch != "y":
                return

        info_reader = InterfaceInfoReader(self.name)

        project_root = info_reader.get_project_root()
        self.implement_datas = info_reader.get_implementation_list()

        import_star_interface_statements = []
        common_import_statements = []
        code_paths = []
        for data in self.implement_datas:
            code_path = data.get("Path", "")
            code_paths.append(code_path)
            if code_path:
                import_interface_statement = get_import_star_statement(
                    project_root, code_path
                )
                if (
                    import_interface_statement
                    and import_interface_statement not in import_star_interface_statements
                ):
                    import_star_interface_statements.append(import_interface_statement)
            codestr = data.get("Implementation", "")
            top_level_functions, class_methods = get_functions_and_class_methods(codestr)
            classes = list(set([cm["class_name"] for cm in class_methods]))
            all_symbols = classes + top_level_functions
            for symbol in all_symbols:
                if symbol.startswith("_"):
                    import_interface_statement = get_import_star_statement(
                        project_root, code_path, symbol
                    )
                    if (
                        import_interface_statement
                        and import_interface_statement
                        not in common_import_statements
                    ):
                        common_import_statements.append(import_interface_statement)
            

        import_str = ""
        if len(common_import_statements) > 0:
            import_str +='try:\n'
            for stmt in common_import_statements:
                import_str += add_python_indent(1, stmt) + "\n"
            import_str +='except Exception as e:\n'
            import_str += add_python_indent(1, f'print(f"[INS_WARN] Failed to import common symbols: {{e}}")') + "\n"
        for idx, stmt in enumerate(import_star_interface_statements):
            module_name = (
                f"{self.name}_"
                + to_valid_module_name(code_path.split(os.sep)[-1])
                + f"_{idx}"
            )
            import_lib_import_str = f"""
import importlib.util
_module_name = "{module_name}"
_file_path = r"{code_path}"
_spec = importlib.util.spec_from_file_location(_module_name, _file_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_module_name] = _mod
_spec.loader.exec_module(_mod)
if hasattr(_mod, "__all__"):
    _names = _mod.__all__
else:
    _names = [name for name in dir(_mod) if not name.startswith("_")]
globals().update({{name: getattr(_mod, name) for name in _names}})
"""
            if has_syntax_error(stmt):
                print(
                    f"[INS_WARN] Syntax error in import statement: {stmt} use importlib instead"
                )
                import_str += import_lib_import_str + "\n"
            else:
                import_str += "try:\n"
                import_str += add_python_indent(1, stmt) + "\n"
                import_str += "except Exception as e:\n"
                import_str += (
                    add_python_indent(
                        1,
                        f'print(f"[INS_WARN] Failed to import module with statement: {{e}}, use importlib instead")',
                    )
                    + "\n"
                )
                import_str += add_python_indent(1, import_lib_import_str) + "\n"

        dead_code_front = f"# {self.name} \n"
        for code_path in code_paths:
            dead_code_front += f"# Source: {code_path}\n"
        dead_code_front += "from Inspection import ENV_BASE\n"
        dead_code_front += f"ENV_DIR = ENV_BASE + '{self.name}/'\n"
        dead_code_front += "from Inspection.adapters import BaseAdapter\n"
        dead_code_front += "import sys\n"
        dead_code_front += "import os\n"
        dead_code_front += f"sys.path.insert(0, '{project_root}')\n"
        dead_code_front += f"os.chdir('{project_root}')\n"
        dead_code_front += f"""
# you can add your custom imports here
{import_str}
# DeadCodeFront end\n
"""
        dead_code_end = f"""
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
if __name__ == "__main__":
    adapter = CustomAdapter()
"""

        customadapter_class_code = self.get_custom_adapter_code_use_template()
        result_code = dead_code_front + customadapter_class_code + dead_code_end
        with open(GEN_PATH + self.name + ".py", "w") as f:
            f.write(result_code)
        self.__create_env_dir()
        print(
            f"[INS_INFO] Adapter {self.name} generated successfully, saved to {GEN_PATH + self.name + '.py'}"
        )

    def __create_env_dir(self):
        env_dir = ENV_BASE + self.name + "/"
        if not os.path.exists(env_dir):
            os.makedirs(env_dir)

    def set_base_ai(self, base_ai: BaseAI):
        self.base_ai = base_ai

    def analyze_project_root_path(self, path):
        abs_path = path
        project_name = self.name

        # 只保留字母和数字进行模糊匹配
        def normalize(s):
            return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

        norm_project = normalize(project_name)
        parts = abs_path.split(os.sep)
        matched_indices = [
            idx for idx, part in enumerate(parts) if norm_project in normalize(part)
        ]
        if matched_indices:
            index = matched_indices[0]
            return os.sep.join(parts[: index + 1])
        else:
            return abs_path

    def get_custom_adapter_code_use_template(self):
        functions = []  # [str]
        classes = []  # [{ 'class_name': str, 'methods': [str] }]

        for data in self.implement_datas:
            code = data.get("Implementation", "")
            temp_functions, temp_classes = get_functions_and_class_methods(code)
            functions.extend(temp_functions)


            # 为每个类添加方法类型信息
            for cls in temp_classes:
                method_types = get_method_types(code, cls["class_name"])
                cls["method_types"] = method_types  # List[{method_name: method_type}]

            self.inherit_info = get_inherit_info(code)

            # 处理继承信息 - 检查基类是否存在于当前类列表中
            if self.inherit_info:
                class_names = [cls["class_name"] for cls in temp_classes]
                # 过滤出基类在当前类列表中的继承关系
                valid_inherit_info = [
                    info for info in self.inherit_info if info[1] in class_names
                ]
                if valid_inherit_info:
                    print(
                        f"[INS_INFO] Found inheritance relationship in {self.name}, processing inheritance information..."
                    )
                    # 为每个继承关系处理基类的方法合并
                    for child_class, parent_class in valid_inherit_info:
                        # 找到父类的方法
                        parent_methods = []
                        for cls in temp_classes:
                            if cls["class_name"] == parent_class:
                                parent_methods = cls["methods"]
                                break
                        # 将父类的方法合并到子类中（如果子类中没有同名方法）
                        for cls in temp_classes:
                            if cls["class_name"] == child_class:
                                for parent_method in parent_methods:
                                    if parent_method not in cls["methods"]:
                                        cls["methods"].append(parent_method)
                                break
                        print(
                            f"[INS_INFO] Merged methods from {parent_class} into {child_class}"
                        )
            classes.extend(temp_classes)

        template_manager = AdapterTemplateManager(self)
        for class_data in classes:
            class_name = class_data["class_name"]
            method_types = class_data.get("method_types", {})
            template_manager.add_class_init(class_name)
            for method_name in class_data["methods"]:
                if method_name == "__init__":
                    continue
                method_type = method_types.get(method_name, "instance")
                template_manager.add_method_mapping(
                    class_name, method_name, method_type
                )
        for function_name in functions:
            template_manager.add_method_mapping(None, function_name)
        return template_manager.generate_class_code()


class AdapterTemplateManager:
    def __init__(self, parent: AdapterGenerator):
        self.classes_to_init: List[str] = []
        self.methods_to_map: List[Dict[str, Any]] = (
            []
        )  # [{'class_name': str, 'method_name': str, 'method_type': str}]
        self.functions: List[str] = []  # 添加函数列表
        self.parent = parent
        self.templates = """
class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        {class_objs_init}


    def create_interface_objects(self, interface_class_name, **kwargs):
        try:
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.func_name = 'create_interface_objects'
            {class_init_if_brances}

        except Exception as e:
            self.result.func_name = 'create_interface_objects'
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {{e}}")

    def run(self, dispatch_key: str, **kwargs):
        try:
            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.func_name = dispatch_key
            {method_mappings_if_brances}

        except Exception as e:
            self.result.func_name = dispatch_key
            self.result.is_success = False
            import traceback
            self.result.fail_reason = str(e) + '\\n' + traceback.format_exc()
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

# DO NOT REMOVE OR MODIFY THE FOLLOWING LINES UNLESS YOU KNOW WHAT YOU ARE DOING
# RegisteredData:
\"\"\"
```json
{registerd_data_json}
```
\"\"\"
"""

    def add_class_init(self, class_name: str):
        if class_name not in self.classes_to_init:
            self.classes_to_init.append(class_name)

    def add_method_mapping(
        self, class_name: str, method_name: str, method_type: str = "instance"
    ):
        if (
            class_name is not None
            and class_name != ""
            and class_name not in self.classes_to_init
        ):
            print(f"[INS_WARN] Class {class_name} not in classes to init")
        mapping = {
            "class_name": class_name,
            "method_name": method_name,
            "method_type": method_type,
        }
        # 检查是否已存在
        existing = any(
            m["class_name"] == class_name and m["method_name"] == method_name
            for m in self.methods_to_map
        )
        if not existing:
            self.methods_to_map.append(mapping)

        # 如果是顶级函数，添加到functions列表
        if class_name is None or class_name == "":
            if method_name not in self.functions:
                self.functions.append(method_name)

    def generate_registered_data_json(self) -> str:
        """生成注册数据的JSON字符串"""
        import json

        # 构建classes数据
        classes_data = []
        inherit_info = self.parent.inherit_info if self.parent.inherit_info else []
        for class_name in self.classes_to_init:
            baseclasses = [info[1] for info in inherit_info if info[0] == class_name]
            # 获取该类的所有方法
            class_methods = []
            for mapping in self.methods_to_map:
                if mapping["class_name"] == class_name:
                    class_methods.append(mapping["method_name"])

            classes_data.append(
                {
                    "name": class_name,
                    "baseclasses": baseclasses,
                    "methods": class_methods,
                }
            )

        # 构建完整的注册数据
        registered_data = {"functions": self.functions, "classes": classes_data}

        return json.dumps(registered_data, indent=4, ensure_ascii=False)

    def generate_class_code(self) -> str:
        # 类对象初始化模板
        class_objs_init = "\n# Class objects\n"
        for class_name in self.classes_to_init:
            class_objs_init += f"self.{class_name.lower()}_obj = None\n"

        # 类初始化分支
        class_init_if_brances = ""
        for i, class_name in enumerate(self.classes_to_init):
            condition_prefix = "if" if i == 0 else "elif"
            class_init_if_brances += f"""
{condition_prefix} interface_class_name == '{class_name}':
    # Create interface object
    self.{class_name.lower()}_obj = {class_name}(**kwargs)
    self.result.interface_return = self.{class_name.lower()}_obj
"""
        if class_init_if_brances == "":
            class_init_if_brances = (
                "\nraise ValueError('No interface classes to initialize')\n"
            )
        else:
            class_init_if_brances += f"""
else:
    # If omitted, create a default interface object
    self.{self.classes_to_init[0].lower()}_obj = {self.classes_to_init[0]}(**kwargs)
    self.result.interface_return = self.{self.classes_to_init[0].lower()}_obj
"""

        # 方法映射分支
        method_mappings_if_brances = ""
        # 为每个类自动添加 __call__ 方法的映射
        for class_name in self.classes_to_init:
            call_mapping = {
                "class_name": class_name,
                "method_name": "__call__",
                "method_type": "instance",
            }
            existing = any(
                m["class_name"] == class_name and m["method_name"] == "__call__"
                for m in self.methods_to_map
            )
            if not existing:
                self.methods_to_map.append(call_mapping)

        for i, mapping in enumerate(self.methods_to_map):
            class_name = mapping["class_name"]
            method_name = mapping["method_name"]
            method_type = mapping.get("method_type", "instance")
            condition_prefix = "if" if i == 0 else "elif"
            if class_name is None or class_name == "":
                # Top-level function
                method_mappings_if_brances += f"""
{condition_prefix} dispatch_key == '{method_name}':
    # Call the {method_name} top-level function
    self.result.interface_return = {method_name}(**kwargs)
    self.result.is_success = True
    self.result.fail_reason = ''
    self.result.func_name = dispatch_key
"""
            else:
                # Class method
                dispatch_key = f"{class_name}_{method_name}"

                # 根据方法类型生成不同的调用代码
                if method_type == "static":
                    # 静态方法：直接通过类名调用
                    method_mappings_if_brances += f"""
{condition_prefix} dispatch_key == '{dispatch_key}':
    # Call the {method_name} static method from {class_name}
    self.result.interface_return = {class_name}.{method_name}(**kwargs)
"""
                elif method_type == "class":
                    # 类方法：直接通过类名调用
                    method_mappings_if_brances += f"""
{condition_prefix} dispatch_key == '{dispatch_key}':
    # Call the {method_name} class method from {class_name}
    self.result.interface_return = {class_name}.{method_name}(**kwargs)
"""
                else:
                    # 实例方法：通过对象实例调用
                    method_mappings_if_brances += f"""
{condition_prefix} dispatch_key == '{dispatch_key}':
    # Call the {method_name} method from {class_name}
    if self.{class_name.lower()}_obj is not None:
        self.result.interface_return = self.{class_name.lower()}_obj.{method_name}(**kwargs)
    elif self.default_obj is not None:
        self.result.interface_return = self.default_obj.{method_name}(**kwargs)
    else:
        raise ValueError(f"Object for class {class_name} not initialized and no default object available.")
"""

        # 添加else分支
        if self.methods_to_map == "":
            method_mappings_if_brances = "pass"
        else:
            method_mappings_if_brances += f"""
else:
    raise ValueError(f"Unknown dispatch key: {{dispatch_key}}")
"""

        # 生成注册数据JSON
        registered_data_json = self.generate_registered_data_json()
        class_code = self.templates.replace(
            "{class_init_if_brances}", add_python_indent(3, class_init_if_brances)
        )
        class_code = class_code.replace(
            "{method_mappings_if_brances}",
            add_python_indent(3, method_mappings_if_brances),
        )
        class_code = class_code.replace("{registerd_data_json}", registered_data_json)
        class_code = class_code.replace(
            "{class_objs_init}", add_python_indent(2, class_objs_init)
        )
        return class_code
