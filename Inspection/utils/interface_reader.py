import json
import os
import re
from Inspection.utils.path_manager import INTERFACE_DOC_PATH , INTERFACE_INFO_PATH , INTERFACE_DATA_PATH
from Inspection.utils.file_lister import FileLister
from Inspection.core.code_processor import has_syntax_error



def replace_relative_imports(code: str, module_abs_path: str, project_root: str) -> str:
    original_code = code
    rel_module_path = os.path.relpath(module_abs_path, project_root).replace(os.sep, "/")
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
        for _ in range(level-1):
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
        print(f"[INS_WARN] Code has syntax errors, cannot replace relative imports: {module_abs_path}")
        return original_code
    return code




class InterfaceDocReader:
    def __init__(self , name):
        self.name = name
        self.__doc = None
        self.__read_doc()
        
    def __read_doc(self):
        try:
            with open(INTERFACE_DOC_PATH + self.name + '.md', 'r') as f:
                self.__doc = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("")
        
    def get_doc(self):
        return self.__doc
    

class InterfaceInfoReader:
    def __init__(self , name):
        self.name = name
        self.info_json = dict()
        self.__load_json()
        self.project_root = self.info_json.get('Project_Root', '')
        self.__preprocess_json()
        
    def __load_json(self):
        try:
            with open(INTERFACE_INFO_PATH + self.name + '.json', 'r') as f:
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
        api_calls = self.info_json.get('API_Calls', [])
        for call in api_calls:
            path = call.get('Path', '')
            if not path:
                print(f"[INS_WARN] Interface call missing path information: {call.get('Name', 'unknown')}")
                continue
            # 替换相对导入
            code = call.get('Code', '')
            if code:
                call['Code'] = replace_relative_imports(code, path, self.project_root) 
        # 替换接口实现中的相对导入
        implementations = self.info_json.get('API_Implementations', [])
        for impl in implementations:
            implementation_code = impl.get('Implementation', '')
            path = impl.get('Path', '')
            if not path:
                print(f"[INS_WARN] Interface implementation missing path information: {impl.get('Name', 'unknown')}")
                continue
            if implementation_code:
                impl['Implementation'] = replace_relative_imports(implementation_code, path, self.project_root)
        
    def get_project_root(self):
        return self.project_root
        
    def print_info(self):
        print(f"[INS_INFO] Interface information:\n{self.info_json}")

    def get_call_list(self):
        return self.info_json.get('API_Calls', [])

    def get_call_dict_by_idx(self , idx:int):
        idx = int(idx)
        api_calls = self.info_json.get('API_Calls', [])
        if idx < 0 or idx >= len(api_calls):
            raise IndexError(f"索引超出范围: {idx}")
        return api_calls[idx]
    
    def get_implementation_list(self):
        return self.info_json.get('API_Implementations', [])

    def get_call_str_by_idx(self , idx:int):
        idx = int(idx)
        api_calls = self.info_json.get('API_Calls', [])
        result_str= ""
        if idx < 0 or idx >= len(api_calls):
            raise IndexError(f"索引超出范围: {idx}")
        i = idx
        name = api_calls[i].get('Name', 'unknown')
        # 去除name中的空行和空格
        name = name.replace('\n', '').replace(' ', '')
        description = api_calls[i].get('Description', 'unknown')
        # 去除description中的所有换行,替换为空格
        description = description.replace('\n', ' ')
        result_str += f"#description: {description}\n"
        result_str += f"#code:\n{api_calls[i]['Code']}\n"
        return result_str
    
    def get_calls(self):
        results = []
        for i in range(len(self.info_json.get('API_Calls', []))):
            result = self.get_call_str_by_idx(i)
            result = '#api_call[' + str(i + 1) + ']:\n' + result
            results.append(result)
        return results

    def get_implementations(self):
        implementations = self.info_json.get('API_Implementations', [])
        results = []
        for i in range(len(implementations)):
            result_str = ""
            name = implementations[i].get('Name', 'unknown')
            # 去除name中的空行
            name = name.replace('\n', ' ')
            description = implementations[i].get('Description', 'unknown')
            # 去除description中的所有换行,替换为空格
            description = description.replace('\n', ' ')
            result_str += f"#api_implementation[{i + 1}]:\n"
            result_str += f"#name: {name}\n"
            result_str += f"#description: {description}\n"
            result_str += f"#implementation:\n{implementations[i]['Implementation']}\n"
            result_str += "\n\'\'\'\n"
            examples = implementations[i].get('Examples', [])
            if examples == []:
                examples = implementations[i].get('Example', [])
            examples_str = str(examples)
            result_str += f"examples:\n {examples_str}\n"
            result_str +="\n\'\'\'\n"
            results.append(result_str)
        return results
    
def expand_json_file():
    pj_names = FileLister(INTERFACE_INFO_PATH, 'json')._file_list
    expanded_pjs_dir = INTERFACE_DATA_PATH + 'InterfaceInfo_expanded'
    for pj_name in pj_names:
        info_dir = expanded_pjs_dir + '/' + pj_name
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)
        inforeader = InterfaceInfoReader(pj_name)
        calls = inforeader.get_call_list()
        implementations = inforeader.get_implementation_list()
        call_data = ''
        impl_data = ''
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
            examples = impl.get('Examples', [])
            if examples == []:
                examples = impl.get('Example', [])
            examples_str = str(examples)
            impl_data += f"examples:\n {examples_str}\n"
            impl_data += "\n'''\n\n\n\n"

        with open(info_dir + '/calls.py', 'w') as f:
            f.write(call_data)
        with open(info_dir + '/implementations.py', 'w') as f:
            f.write(impl_data)
        with open(info_dir + '/root.txt', 'w') as f:
            f.write(inforeader.get_project_root())

if __name__ == "__main__":
    print("Expanding interface information file...")
    expand_json_file()