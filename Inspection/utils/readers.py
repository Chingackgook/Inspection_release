import json
import os
import re
from typing import List, Dict
from Inspection.utils.path_manager import (
    INTERFACE_DOC_PATH,
    INTERFACE_INFO_PATH,
    INTERFACE_DATA_PATH,
    CUSTOM_ADAPTER_PATH,
    SIMULATOR_PATH,
    DUMB_SIMULATOR_PATH,
    ENV_BASE,
)
from Inspection.utils.file_lister import FileLister
from Inspection.core.code_processor import has_syntax_error, extract_code
from Inspection.utils.config import CONFIG


class InterfaceDocReader:
    def __init__(self, name):
        self.name = name
        self.__doc = None
        self.__read_doc()

    def __read_doc(self):
        try:
            with open(INTERFACE_DOC_PATH + self.name + ".md", "r") as f:
                self.__doc = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("")

    def get_doc(self):
        return self.__doc

    def get_doc_with_api(self, api_name, base_ai):
        """使用AI将相关的接口文档提取出来"""
        force_regenerate = CONFIG.get("force_regenerate", False)
        env_dir = ENV_BASE + f"{self.name}/"
        doc_cache_path = env_dir + "cache/" + api_name + "_doc.txt"
        if not force_regenerate:
            # 检查是否已经存在提取的文档
            if os.path.exists(doc_cache_path):
                with open(doc_cache_path, "r") as f:
                    doc = f.read()
                print(
                    f"[INS_INFO] Extracted documentation cache already exists: `{doc_cache_path}`, using cache"
                )
                return doc
        ai = base_ai.copy()
        prompt = f"""
{self.__doc}
For the above interface documentation, please extract the parts related to the {api_name} interface, making sure to retain relevant contextual key information.
"""
        response = ai.generate_text(prompt)
        if not os.path.exists(os.path.dirname(doc_cache_path)):
            os.makedirs(os.path.dirname(doc_cache_path))
        with open(doc_cache_path, "w") as f:
            f.write(response)
        return response


class InterfaceInfoReader:
    def __init__(self, name: str):
        self.name = name
        self.info_json = dict()
        self.__load_json()
        self.project_root = self.info_json.get("Project_Root", "")
        self.__preprocess_json()

    def __load_json(self):
        try:
            with open(INTERFACE_INFO_PATH + self.name + ".json", "r") as f:
                self.info_json = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("")
        except json.JSONDecodeError:
            print(f"[INS_ERR] JSON decoding error: {self.name}.json")
            raise ValueError(f"JSON解码错误: {self.name}.json")

    def __preprocess_json(self):
        """
        根据项目跟目录以及目标模块目录，将接口实现以及接口调用的所有相对导包转换为绝对导包
        """
        # 替换接口调用中的相对导入
        api_calls = self.info_json.get("API_Calls", [])
        for call in api_calls:
            path = call.get("Path", "")
            if not path:
                print(
                    f"[INS_WARN] Interface call missing path information: {call.get('Name', 'unknown')}"
                )
                continue
            # 替换相对导入
            code = call.get("Code", "")
            if code:
                call["Code"] = replace_relative_imports(code, path, self.project_root)
        # 替换接口实现中的相对导入
        implementations = self.info_json.get("API_Implementations", [])
        for impl in implementations:
            implementation_code = impl.get("Implementation", "")
            path = impl.get("Path", "")
            if not path:
                print(
                    f"[INS_WARN] Interface implementation missing path information: {impl.get('Name', 'unknown')}"
                )
                continue
            if implementation_code:
                impl["Implementation"] = replace_relative_imports(
                    implementation_code, path, self.project_root
                )

    def get_project_root(self):
        return self.project_root

    def print_info(self):
        print(f"[INS_INFO] Interface information:\n{self.info_json}")

    def get_call_list(self):
        return self.info_json.get("API_Calls", [])

    def get_call_dict_by_idx(self, idx: int) -> Dict:
        idx = int(idx)
        api_calls = self.info_json.get("API_Calls", [])
        if idx < 0 or idx >= len(api_calls):
            raise IndexError(f"索引超出范围: {idx}")
        return api_calls[idx]

    def get_implementation_list(self) -> List[Dict]:
        return self.info_json.get("API_Implementations", [])

    def get_implementation_dict_by_idx(self, idx: int):
        idx = int(idx)
        implementations = self.info_json.get("API_Implementations", [])
        if idx < 0 or idx >= len(implementations):
            raise IndexError(f"索引超出范围: {idx}")
        return implementations[idx]

    def get_call_str_by_idx(self, idx: int):
        idx = int(idx)
        api_calls = self.info_json.get("API_Calls", [])
        result_str = ""
        if idx < 0 or idx >= len(api_calls):
            raise IndexError(f"索引超出范围: {idx}")
        i = idx
        name = api_calls[i].get("Name", "unknown")
        # 去除name中的空行和空格
        name = name.replace("\n", "").replace(" ", "")
        description = api_calls[i].get("Description", "unknown")
        # 去除description中的所有换行,替换为空格
        description = description.replace("\n", " ")
        result_str += f"#description: {description}\n"
        result_str += f"#code:\n{api_calls[i]['Code']}\n"
        return result_str

    def get_calls(self):
        results = []
        for i in range(len(self.info_json.get("API_Calls", []))):
            result = self.get_call_str_by_idx(i)
            result = "#api_call[" + str(i + 1) + "]:\n" + result
            results.append(result)
        return results

    def get_implementations(self):
        implementations = self.info_json.get("API_Implementations", [])
        results = []
        for i in range(len(implementations)):
            result_str = ""
            name = implementations[i].get("Name", "unknown")
            # 去除name中的空行
            name = name.replace("\n", " ")
            description = implementations[i].get("Description", "unknown")
            # 去除description中的所有换行,替换为空格
            description = description.replace("\n", " ")
            result_str += f"#api_implementation[{i + 1}]:\n"
            result_str += f"#name: {name}\n"
            result_str += f"#description: {description}\n"
            result_str += f"#implementation:\n{implementations[i]['Implementation']}\n"
            result_str += "\n'''\n"
            examples = implementations[i].get("Examples", [])
            if examples == []:
                examples = implementations[i].get("Example", [])
            examples_str = str(examples)
            result_str += f"examples:\n {examples_str}\n"
            result_str += "\n'''\n"
            results.append(result_str)
        return results


class ProjectRegistrationInfoReader:
    def have_interface_info(self) -> list:
        path = INTERFACE_INFO_PATH
        pj_names = FileLister(path, "json")._file_list
        return pj_names

    def have_interface_doc(self) -> bool:
        path = INTERFACE_DOC_PATH
        pj_names = FileLister(path, "md")._file_list
        return pj_names

    def have_adapter(self) -> bool:
        path = CUSTOM_ADAPTER_PATH
        pj_names = FileLister(path, "py")._file_list
        return pj_names

    def have_both_info_doc_adapter(self) -> list:
        info_pjs = self.have_interface_info()
        doc_pjs = self.have_interface_doc()
        adapter_pjs = self.have_adapter()
        both_pjs = []
        for pj in info_pjs:
            if pj in doc_pjs and pj in adapter_pjs:
                both_pjs.append(pj)
        return both_pjs

    def have_simulator(self) -> list:
        dirs = FileLister(SIMULATOR_PATH, "dir", not_include="backup_")._file_list
        info_pjs = self.have_interface_info()
        doc_pjs = self.have_interface_doc()
        adapter_pjs = self.have_adapter()
        simulator_pjs = []
        for pj in dirs:
            if pj in info_pjs and pj in doc_pjs and pj in adapter_pjs:
                simulator_pjs.append(pj)
        return simulator_pjs

    def have_dumb_simulator(self) -> list:
        dirs = FileLister(DUMB_SIMULATOR_PATH, "dir", not_include="backup_")._file_list
        info_pjs = self.have_interface_info()
        doc_pjs = self.have_interface_doc()
        adapter_pjs = self.have_adapter()
        dumb_simulator_pjs = []
        for pj in dirs:
            if pj in info_pjs and pj in doc_pjs and pj in adapter_pjs:
                dumb_simulator_pjs.append(pj)
        return dumb_simulator_pjs

    def can_generate_doc(self, pj_name: str) -> bool:
        if pj_name in self.have_interface_info():
            return True
        return False

    def can_generate_adapter(self, pj_name: str) -> bool:
        if pj_name in self.have_interface_info():
            return True
        return False

    def can_generate_simulator(self, pj_name: str) -> bool:
        if pj_name in self.have_adapter():
            if pj_name in self.have_interface_info():
                if pj_name in self.have_interface_doc():
                    return True
        return False

    def can_generate_dumb_simulator(self, pj_name: str) -> bool:
        return self.can_generate_simulator(pj_name)

    def get_adapter_registed_apis(self, pj_name: str) -> List[str]:
        if pj_name not in self.have_adapter():
            raise ValueError(f"Project '{pj_name}' does not have a registered adapter.")
        adapter_path = CUSTOM_ADAPTER_PATH + pj_name + ".py"
        try:
            with open(adapter_path, "r") as f:
                adapter_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Adapter file for project '{pj_name}' not found.")
        registerd_data_json = json.loads(extract_code(adapter_code, "json"))
        avaliable_funcs = registerd_data_json.get("functions", []).copy()
        classes = registerd_data_json.get("classes", []).copy()
        for the_class in classes:
            methods = the_class.get("methods", [])
            for method in methods:
                name = f"{the_class['name']}_{method}"
                avaliable_funcs.append(name)
        return avaliable_funcs, registerd_data_json


def replace_relative_imports(code: str, module_abs_path: str, project_root: str) -> str:
    original_code = code
    rel_module_path = os.path.relpath(module_abs_path, project_root).replace(
        os.sep, "/"
    )
    module_dir = os.path.dirname(rel_module_path)
    module_package = module_dir.replace("/", ".").strip(".")
    pattern = re.compile(r"(from\s+)(\.+)([\w\.]*)(\s+import\s+[^;\n]+)", re.MULTILINE)

    def repl(match):
        from_part, dots, module_path, import_part = match.groups()
        level = len(dots)
        if module_package:
            parent_parts = module_package.split(".")
        else:
            parent_parts = []
        # 检查层级是否超出范围
        if level > len(parent_parts):
            # 如果层级超出，返回原始匹配
            return match.group(0)
        # 计算目标包路径
        for _ in range(level - 1):
            if parent_parts:
                parent_parts.pop()
        if module_path:
            absolute_module = ".".join(parent_parts + module_path.split("."))
        else:
            absolute_module = ".".join(parent_parts)
        result = f"{from_part}{absolute_module}{import_part}"
        return result

    code = pattern.sub(repl, code)
    if has_syntax_error(code):
        return original_code
    return code


def expand_json_file():
    # 将接口信息的json文件展开为py文件，方便查看，不是项目的核心功能

    pj_names = FileLister(INTERFACE_INFO_PATH, "json")._file_list
    expanded_pjs_dir = INTERFACE_DATA_PATH + "InterfaceInfo_expanded"
    for pj_name in pj_names:
        info_dir = expanded_pjs_dir + "/" + pj_name
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        inforeader = InterfaceInfoReader(pj_name)
        calls = inforeader.get_call_list()
        implementations = inforeader.get_implementation_list()
        call_data = ""
        impl_data = ""
        for call in calls:
            call_data += f"###api_call[{calls.index(call) + 1}]:\n"
            call_data += f"###name: {call.get('Name', 'unknown')}\n"
            call_data += f"###description: {call.get('Description', 'unknown')}\n"
            call_data += f"###path: {call.get('Path', '')}\n"
            call_data += f"###code:\n\n\n\n{call.get('Code', '')}\n"
            call_data += "\n\n"

        for impl in implementations:
            impl_data += f"###api_implementation[{implementations.index(impl) + 1}]:\n"
            impl_data += f"###name: {impl.get('Name', 'unknown')}\n"
            impl_data += f"###description: {impl.get('Description', 'unknown')}\n"
            impl_data += f"###path: {impl.get('Path', '')}\n"
            impl_data += f"###implementation:\n\n\n\n{impl.get('Implementation', '')}\n"
            impl_data += "\n'''"
            examples = impl.get("Examples", [])
            if examples == []:
                examples = impl.get("Example", [])
            examples_str = str(examples)
            impl_data += f"examples:\n {examples_str}\n"
            impl_data += "\n'''\n\n\n\n"

        with open(info_dir + "/calls.py", "w") as f:
            f.write(call_data)
        with open(info_dir + "/implementations.py", "w") as f:
            f.write(impl_data)
        with open(info_dir + "/root.txt", "w") as f:
            f.write(inforeader.get_project_root())


if __name__ == "__main__":
    print("Expanding interface information file...")
    expand_json_file()
