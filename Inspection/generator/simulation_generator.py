# 模拟执行代码生成器
from Inspection.ai.base_ai import BaseAI
from Inspection.utils.path_manager import BASE_DIR
from Inspection.core.code_processor import *
from Inspection.utils.readers import (
    InterfaceInfoReader,
    InterfaceDocReader,
    ProjectRegistrationInfoReader,
)
import os
import json
import time
from Inspection.utils.path_manager import SIMULATOR_PATH
from Inspection.utils.config import CONFIG
from Inspection.utils.tools import get_available_outter_resources_exts


class SimulationGenerator:
    def __init__(self, name: str):
        self.name = name
        self.ai: BaseAI = None
        self.prompt_user_for_overwrite = CONFIG.get("ask", True)  # 询问是否覆盖
        self.simulation_path = SIMULATOR_PATH + self.name + "/"
        self.analysis_data = ""
        self.call_idx = 0
        self.analysis_temprature = CONFIG.get(
            "simulation_analysis_temprature", 0.5
        )  # 分析温度
        self.generate_code_temprature = CONFIG.get(
            "simulation_generate_code_temprature", 0.3
        )  # 生成代码温度

    def set_base_ai(self, base_ai_instance):
        self.ai = base_ai_instance

    def _write_analysis_data(self):
        with open(self.analysis_data_path, "w") as f:
            f.write(self.analysis_data)

    def generate_simulation(self, api_name: str, call_idx: int):
        if not ProjectRegistrationInfoReader().can_generate_simulator(self.name):
            print(
                f"[INS_ERROR] Project {self.name} can not generate dumb simulator, check whether the interface info, interface doc, and adapter all exist"
            )
            return
        self.api_name = api_name
        # 生成根据调用信息，以及原调用步骤生成的模拟执行代码
        simulation_path = self.simulation_path
        start_time = time.time()
        info_reader = InterfaceInfoReader(self.name)
        call_data = info_reader.get_call_str_by_idx(call_idx)
        self.call_idx = call_idx
        call_path = info_reader.get_call_dict_by_idx(call_idx).get("Path", "")
        future_import_data, import_data = extract_import_statements(call_data)
        self.analysis_data_path = (
            simulation_path + f"call{call_idx}_{self.api_name}_simulator_analysis.md"
        )
        self.write_path = (
            simulation_path + f"call{call_idx}_{self.api_name}_simulator.py"
        )
        # 检查文件是否已经存在模拟执行代码
        if (
            os.path.exists(self.write_path)
            and self.prompt_user_for_overwrite
        ):
            print(
                f"[INS_WARN] Static executable artifact {self.name} for call index {call_idx} already exists at {self.write_path}."
            )
            ch = input("Overwrite? (y/n)")
            if ch != "y":
                return
        # 创建目录
        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)
        self.analysis_data = ""  # 用于存储分析结果，最后会写入记录md文件中
        print(
            "[INS_INFO] Analysis process located in "
            + self.analysis_data_path
            + " for viewing anytime"
        )
        code = call_data
        self.doc = ''
        ONLY_REPLACE_EXE = CONFIG.get("only_replace_exe_in_simulation_gen", False)
        # 以上是该方法的一些准备工作，下面开始进行ai生成模拟执行代码
        if not ONLY_REPLACE_EXE:
            self.doc = InterfaceDocReader(self.name).get_doc_with_api(api_name, self.ai)

        if not ONLY_REPLACE_EXE:
            code = self.change_code_to_executable(code)

        if not code:
            print(
                f"[INS_ERR] Static executable artifact generation failed, unable to convert original call to static executable artifact"
            )
            return

        code = remove_definitions_by_names(
            code, extract_from_import_object(import_data)
        )
        code, replace_type = self.change_interface_to_exe_methods_withAST(code)

        if replace_type["type"] == "class_method" and not ONLY_REPLACE_EXE:
            code = self.check_and_add_class_init_with_ai(
                code, replace_type["class_name"]
            )

        code = remove_future_imports(code)
        code = replace_file_variable_in_code(code, call_path)

        code = (
            f"""
{future_import_data}
import sys
from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.{self.name} import *
exe = Executor('{self.name}','simulation')
OUTPUTS = exe.now_record_path
sys.argv[0] = '{call_path}'
sys.path.append('{os.path.dirname(call_path)}')
{import_data}
# End
# Please DO NOT modify the code above this line.\n\n
"""
            + code
        )
        if not ONLY_REPLACE_EXE:
            code = self.optimize_code(code)
            code = self.change_external_resources_path(code)

        code = remove_assignments("exe.now_record_path", code, use_regex=True)
        total_time = time.time() - start_time
        self.analysis_data += f"\n\n\n$$$$$静态可执行工件生成耗时$$$$$\n"
        self.analysis_data += f"Total time: {total_time:.2f} seconds\n"
        self._write_analysis_data()

        self.write_code_to_file(code)

    def write_code_to_file(self, code: str):
        with open(self.write_path, "w") as f:
            f.write(code)
        print(
            f"[INS_INFO] Static executable artifact {self.name} generated successfully, saved to {self.write_path}"
        )

    def change_code_to_executable(self, code: str):
        ai = self.ai.copy()
        print("[INS_INFO] Analyzing code logic and converting to executable code ...")
        analysis_prompt = f"""
Here is a piece of code that calls an intelligent module:
```python
{code}
```
Here is the API documentation for the key functions/methods:
```api documentation:
{self.doc}
```
Please explain in detail what the main execution logic of this code is. Provide a thorough analysis.
"""
        self.analysis_data += "$$$$$代码逻辑分析$$$$$\n"
        self.analysis_data += ai.generate_text(
            analysis_prompt, temperature=self.analysis_temprature
        )
        self._write_analysis_data()
        suppose_prompt = f"""
Based on the above analysis, if we directly run this piece of code using Python’s `exec` function, what potential problems might occur?
How should this code be modified, with minimal changes to its logic, so that it can be executed directly via the `exec` function.
Generate a plan for modifying this code (do not generate the modified code yet).
Tips:
1. Remove or replace any interactive input mechanisms (such as user runtime interaction, argparse, typer_app, https request, or web UI interactions) with hardcoded values and avoid dead loops. 
Especially for the input, if it has a default value, please use the default value given in the call data. If the default value or a possible value can be inferred from the source code, use that value. If no information can be inferred from the source code, use a placeholder path like 'path/to/...'. 
For all occurrences of apikey and apibase with values retrieved from environment variables using os.environ, assume that all environment variables already exist and replace them directly with their values. Do not use placeholders for both os.environ and other environment variable retrieval methods.
Use default values in the original code for other parameters wherever possible.
Do not define mock classes or functions, please use the actual implementations.
Avoid using namspace, use simple variable assignments or dictionaries instead.
2. If the code is a Python module without an `if __name__ == "__main__"` block or any other execution entry point, you need to add an entry point and provide appropriate input data. You need to ensure that the code is able to execute the {self.api_name} function/method correctly
Please analyze first — do not generate the modified code yet.
"""
        self.analysis_data += "\n\n\n$$$$$代码执行补全分析$$$$$\n"
        self.analysis_data += ai.generate_text(
            suppose_prompt, temperature=self.analysis_temprature
        )
        self._write_analysis_data()
        generate_prompt = f"""
{code}

Based on the analysis above, modify this piece of code so that it can be directly executed using the `exec` function.

Requirements:
1. The generated code must be complete Python code that can be run directly via `exec`.
2. Keep the original logic structure as intact as possible; Avoid leaving out any parts of the original code; Try to minimize changes to the original code .
3. Output **only** the Python code, wrapped in ```python ... ```, with no extra explanations or text.

"""
        self.analysis_data += "\n\n\n$$$$$代码执行补全结果$$$$$\n"
        result = ai.generate_text(
            generate_prompt, temperature=self.generate_code_temprature
        )
        self.analysis_data += result
        self._write_analysis_data()
        result = extract_python_code(result)
        return result

    def change_interface_to_exe_methods_withAST(self, code: str):
        """
        将代码中的接口调用替换为exe.run("function_name" , **kwargs)的形式
        :param code: Python代码字符串
        :return: 替换后的代码字符串
        """
        print("[INS_INFO] Replacing interface calls with exe.run form ...")
        origin_code = code
        _, registerd_data_json = (
            ProjectRegistrationInfoReader().get_adapter_registed_apis(self.name)
        )
        top_level_functions = registerd_data_json.get("functions", [])
        classes = registerd_data_json.get("classes", [])
        inforeader = InterfaceInfoReader(self.name)
        impl_code = inforeader.get_implementation_dict_by_idx(self.call_idx).get(
            "Implementation", ""
        )
        # process top_level_functions
        if self.api_name in top_level_functions:
            func_name = self.api_name
            position_args, _ = get_function_param_names(impl_code, func_name)
            code = replace_call_with_new_method(
                code,
                positionlist=position_args,
                preobj=None,  # 表示替换顶级函数
                premethod=func_name,
                newobj="exe",
                newmethod="run",
                first_arg=f"'{func_name}'",
            )
            code = replace_call_with_new_method(
                code,
                positionlist=position_args,
                preobj=extract_import_objects(
                    code
                ),  # 解决 import a // res = a.old_func() 这种调用
                premethod=func_name,
                newobj="exe",
                newmethod="run",
                first_arg=f"'{func_name}'",
            )
            return code, {
                "type": "top_level_function",
                "class_name": None,
                "method": func_name,
            }

        else:
            # process class methods
            cls = None
            for c in classes:
                if self.api_name.startswith(c.get("name", "")):
                    cls = c
                    break
            if cls is None:
                print(
                    f"[INS_ERR] API {self.api_name} not found in registered functions or classes"
                )
                return code, {"type": "not_found", "class_name": None, "method": None}
            class_name = cls.get("name", "")
            baseclasses = cls.get("baseclasses", [])
            instance_names = get_class_instance_names(code, class_name)
            for baseclass in baseclasses:
                base_instance_names = get_class_instance_names(code, baseclass)
                instance_names.extend(
                    name for name in base_instance_names if name not in instance_names
                )
            method_list = cls.get("methods", [])

            # dispath_obj_calls 形如 obj()
            if self.api_name == f"{class_name}___call__":
                method_name = "__call__"
                call_position_args, _ = get_function_param_names(
                    impl_code, "__call__", class_name, baseclasses
                )
                for instance_name in instance_names:
                    code = replace_call_with_new_method(
                        code,
                        positionlist=call_position_args,
                        preobj=None,  # 表示替换原始对象调用
                        premethod=instance_name,  # 对象名作为方法名
                        newobj="exe",
                        newmethod="run",
                        first_arg=f"'{class_name}___call__'",
                    )

            # dispatch_methods
            else:
                method_name = None
                for name in method_list:
                    if self.api_name == f"{class_name}_{name}":
                        method_name = name
                        break
                if method_name is None:
                    print(
                        f"[INS_ERR] Method {self.api_name} not found in class {class_name}"
                    )
                    return code, {
                        "type": "not_found",
                        "class_name": class_name,
                        "method": None,
                    }
                print(
                    f"[INS_INFO] Replacing method call {method_name} of class {class_name} with exe.run form"
                )
                dispatch_key = f"{class_name}_{method_name}"
                position_args, _ = get_function_param_names(
                    impl_code, method_name, class_name, baseclasses
                )

                code = replace_call_with_new_method(
                    code,
                    positionlist=position_args,
                    preobj=instance_names,  # 对象名列表
                    premethod=method_name,
                    newobj="exe",
                    newmethod="run",
                    first_arg=f"'{dispatch_key}'",
                )

            init_position_args, _ = get_function_param_names(
                impl_code, "__init__", class_name, baseclasses
            )
            _ , classmethods = get_functions_and_class_methods(origin_code)
            class_defs = set()
            for cm in classmethods:
                class_defs.add(cm["class_name"])

            if class_name not in class_defs:
                code = replace_call_with_new_method(
                    code,
                    positionlist=init_position_args,
                    preobj=None,  # 表示替换构造函数，即类似于顶级函数的替换
                    premethod=class_name,  # 构造函数名即类名
                    newobj="exe",
                    newmethod="create_interface_objects",
                    first_arg=f"'{class_name}'",  # 第一个参数为类名
                )
            else:
                code = append_call_with_new_method(
                    code,
                    positionlist=init_position_args,
                    preobj=None,  # 表示替换构造函数，即类似于顶级
                    premethod=class_name,  # 构造函数名即类名
                    newobj="exe",
                    newmethod="create_interface_objects",
                    first_arg=f"'{class_name}'",  # 第一个参数为类名
                )

            return code, {
                "type": "class_method",
                "class_name": class_name,
                "method": method_name,
            }

    def check_and_add_class_init_with_ai(self, code: str, class_name: str):
        """
        检查代码中是否有类的实例化，如果没有，则添加实例化代码
        """
        if (
            code.find(f"exe.create_interface_objects('{class_name}'") != -1
            or code.find(f'exe.create_interface_objects("{class_name}"') != -1
        ):
            # 已经有实例化代码
            return code
        prompt = f"""
For the following Python source code:
{code}

In the code, there might be a factory function or method to create an instance of the `{class_name}` class and assign it to a variable.  
Please find this assignment statement. Then, immediately on the next line, add a line of code to also assign this newly created instance to `exe.adapter.default_obj`.

For example, if the code contains:
`my_object = create_my_class_instance()`
The modified code should be:
`my_object = create_my_class_instance()`
`exe.adapter.default_obj = my_object`

Modification requirements:
1. Please return only the modified complete Python code, wrapped in ```python ... ```.
2. Make minimal changes to ensure the original logic of the code remains intact.
"""
        ai_instance = self.ai.copy()
        print("[INS_INFO] adding class instantiation code with AI ...")
        result = ai_instance.generate_text(
            prompt, temperature=self.generate_code_temprature
        )
        code = extract_python_code(result)
        return code

    def change_external_resources_path(self, code: str):
        """
        将一些图片，音频，视频等外部资源的路径替换为本项目默认静态资源
        """
        ai_instance = self.ai.copy()
        print(
            "[INS_INFO] Attempting to replace external resource paths with default static resources ..."
        )
        ask_prompt = f"""
Here is a piece of Python code:
{code}
Please analyze whether there are placeholder paths in this code that contain "path/to" or similar placeholder patterns.
Focus only on variables or dictionary values that contain placeholder paths like:
- "path/to/image.jpg"
- "path/to/audio.mp3" 
- "path/to/video.mp4"
- "path/to/text.txt"
- "path/to/some_file"
- similar placeholder patterns

For each placeholder path found, determine:
1. Whether it should correspond to a single file or a folder
2. Whether it's an image, audio, video, or text file based on the context or file extension
3. The corresponding variable names or python dictionary keys in the code ()
4. The placeholder value (the right side of the assignment statement)

Only analyze paths that are clearly placeholders, ignore real file paths or paths that don't contain placeholder patterns.
Classify the placeholder resources into four categories: images, audios, videos, texts.(pdf file will be treated as images)
"""
        self.analysis_data += "\n\n\n$$$$$External Resource Path Analysis$$$$$\n"
        self.analysis_data += ai_instance.generate_text(
            ask_prompt, temperature=self.analysis_temprature
        )
        self._write_analysis_data()

        format_prompt = r"""
For the placeholder paths identified above (only those containing "path/to" or similar placeholder patterns), please return in the following JSON format:
```json
{
    "images": [
        {
            "name": "some_img",
            "is_folder": false,
            "value": "path/to/image.jpg",
            "suffix": "jpg"
        },
        {
            "name": "some_pdf_path",
            "is_folder": false,
            "value": "path/to/some_file.pdf",
            "suffix": "pdf"
        }
    ],
    "audios": [
        {
            "name": "some_audio", 
            "is_folder": true,
            "value": "path/to/audios/",
            "suffix": ""
        }
    ],
    "videos": [
        {
            "name": "some_video_path",
            "is_folder": false,
            "value": "path/to/video.mp4", 
            "suffix": "mp4"
        }
    ],
    "texts": [
        {
            "name": "some_text_path",
            "is_folder": false,
            "value": "path/to/document.txt",
            "suffix": "txt"
        }
    ]
}
```
Please note:
1. ONLY include variables/paths that contain placeholder patterns like "path/to"
2. Do not include real file paths or existing file references
3. The returned JSON format must strictly follow the above format
4. Variable names and paths must be string types
5. If there are no placeholder resources of a certain type, return an empty list for that field

Where `name` is the corresponding variable name or dictionary key in the code, `value` is the placeholder path (the right side of the assignment statement as a string).
`is_folder` indicates whether the path should be a file or a folder.
`suffix` indicates the file extension (empty string if it's a folder).
Please wrap the returned JSON content with ```json ...```
"""
        self.analysis_data += "\n\n\n$$$$$External Resource Path Format Analysis$$$$$\n"
        analysis_result = ai_instance.generate_text(format_prompt, temperature=0)
        self.analysis_data += analysis_result
        self._write_analysis_data()
        try:
            resource_info = extract_code(analysis_result, "json")
            resource_info = json.loads(resource_info)
        except json.JSONDecodeError:
            print(
                f"[INS_ERR] External resource path analysis result parsing failed, please check if the returned json format is correct"
            )
            return code

        # 接下来使用正则表达式将代码中的路径替换为默认静态资源路径
        from Inspection.core.code_processor import replace_assignment_which_with_path_to

        images_list = resource_info.get("images", [])
        audios_list = resource_info.get("audios", [])
        videos_list = resource_info.get("videos", [])
        texts_list = resource_info.get("texts", [])

        def need_replace(value: str):
            if value == None:
                return False
            try:
                if (
                    value.find("output") != -1
                    or value.find("result") != -1
                    or value.find("save") != -1
                ):
                    # 如果路径中包含output或result，表示是输出结果路径
                    return False
                if value.find("path") != -1 and value.find("to") != -1:
                    # 这种情况往往表示这是一个占位符路径
                    return True
                if value.find("OUTPUTS") != -1 or value.find("ENV_DIR") != -1:
                    # 如果路径中包含OUTPUTS，表示已经出现幻觉了，需要替换
                    return True
                return False
            except Exception as e:
                print(
                    f"[INS_WARN] Error occurred while checking if path needs replacement: {e}"
                )
                return False

        def _replace_resource_paths(
            code, resource_list, resource_type, default_suffix, folder_name, file_name
        ):
            for item_data in resource_list:
                variable_name = item_data.get("name", "")
                if not need_replace(item_data.get("value", "")):
                    continue
                if item_data.get("is_folder", False):
                    new_value = f"RESOURCES_PATH + '{resource_type}/{folder_name}'"
                else:
                    suffix = item_data.get("suffix", "") or default_suffix
                    avliavle_list = get_available_outter_resources_exts(resource_type)
                    if suffix not in avliavle_list:
                        suffix = default_suffix
                    new_value = (
                        f"RESOURCES_PATH + '{resource_type}/{file_name}.{suffix}'"
                    )

                code = replace_assignment_which_with_path_to(
                    code, variable_name, new_value
                )
                code = replace_dict_value(code, variable_name, new_value)
            return code

        code = _replace_resource_paths(
            code, images_list, "images", "jpg", "default_images_folder", "default"
        )
        code = _replace_resource_paths(
            code, audios_list, "audios", "wav", "default_audios_folder", "default"
        )
        code = _replace_resource_paths(
            code, videos_list, "videos", "mp4", "default_videos_folder", "default"
        )
        code = _replace_resource_paths(
            code, texts_list, "texts", "jsonl", "default_texts_folder", "default"
        )

        code = clean_path_to_in_code(code)  # 清理代码中的path/to占位符
        return code

    def optimize_code(self, source_code: str):
        ai_instance = self.ai.copy()
        print("[INS_INFO] Optimizing ...")
        optimization_int_prompt = f"""
Here is a piece of Python code:
{source_code}
Please find the places in this code where files are **Final** output, please tell me the variable names of the output files (Do not include any input files). 
please answer the variable names in a list format wrapped in ```list ... ```
e.g. 
```list
['output_path_1', 'output_path_2']
```
if there are no output files, please return an empty list.
"""
        analysis_data = ai_instance.generate_text(
            optimization_int_prompt, temperature=self.analysis_temprature
        )
        output_files_list = extract_code(analysis_data, "list", first_only=True)
        if has_syntax_error(output_files_list):
            output_files_list = []
        try:
            output_files_list = eval(output_files_list) if output_files_list else []
        except:
            output_files_list = [str(output_files_list).strip("[]")]
        if not isinstance(output_files_list, (list, tuple)):
            output_files_list = []
        for output_file_var in output_files_list:
            if output_file_var == 'OUTPUTS':
                output_files_list.remove(output_file_var)
        output_files_list_str = ", ".join(output_files_list)
        self.analysis_data += "\n\n\n$$$$$Code Optimization Analysis$$$$$\n"
        self.analysis_data += analysis_data
        self._write_analysis_data()
        optimization_prompt = "Please optimize the source code:\n"
        q1 = f"""
Task 1: please replace the **final output** file `{output_files_list_str}` root paths with an existing global variable OUTPUTS 
(already exists, no need to define, type is string)
"""
        q2 = f"""
Task 2: please check for simple syntax errors in this code and fix them. 
If it uses unittest to run the main logic, please remove the unittest code and change it to run the main logic directly.
Please preserve the original code structure and logic as much as possible, and you can add necessary comments. Only return the modified code.
Wrap the generated code with ```python ... ```
"""
        if len(output_files_list) > 0:
            optimization_prompt += q1 + "\n"
        optimization_prompt += q2 + "\n"
        result = ai_instance.generate_text(
            optimization_prompt, temperature=self.generate_code_temprature
        )
        self.analysis_data += "\n\n\n$$$$$Code Optimization Result$$$$$\n"
        self.analysis_data += result
        self._write_analysis_data()
        optimized_code = extract_python_code(result)
        return optimized_code


