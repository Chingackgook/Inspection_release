import time
import re
import os
import ast

from Inspection.ai.base_ai import BaseAI
from Inspection.generator.injector import Injector
from Inspection.utils.readers import (
    InterfaceDocReader,
    InterfaceInfoReader,
    ProjectRegistrationInfoReader,
)
from Inspection.core.code_processor import (
    extract_python_code,
    remove_assignments,
    extract_import_statements,
)
from Inspection.core.code_processor import (
    remove_definitions_by_names,
    extract_from_import_object,
    remove_function_calls,
    remove_imports_from_code,
)
from Inspection.utils.path_manager import BASE_DIR
from Inspection.utils.config import CONFIG
from concurrent.futures import ThreadPoolExecutor
from Inspection.utils.path_manager import SIMULATOR_PATH, DUMB_SIMULATOR_PATH


def extract_arg_names_for_apiname(code, target_apiname):
    tree = ast.parse(code)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "exe"
                and node.func.attr == "run"
            ):
                if node.args and (
                    (
                        hasattr(ast, "Constant")
                        and isinstance(node.args[0], ast.Constant)
                        and node.args[0].value == target_apiname
                    )
                    or (
                        isinstance(node.args[0], ast.Str)
                        and node.args[0].s == target_apiname
                    )
                ):
                    arg_names = []
                    # 处理位置参数中的 *args（从第二个参数开始，第一个是 target_apiname）
                    for i, arg in enumerate(node.args[1:], 1):
                        if isinstance(arg, ast.Starred):
                            # 处理 *args 情况
                            if isinstance(arg.value, ast.Name):
                                arg_names.append(arg.value.id)

                    # 处理关键字参数
                    for kw in node.keywords:
                        if kw.arg is not None:
                            # 普通关键字参数
                            arg_names.append(kw.arg)
                        else:
                            # 处理 **kwargs 情况
                            if isinstance(kw.value, ast.Name):
                                arg_names.append(kw.value.id)

                    result.append(arg_names)
    if not result:
        raise ValueError(f"未找到exe.run('{target_apiname}', ...)调用参数")
    r = result[0]
    if not r:
        raise ValueError(f"未找到exe.run('{target_apiname}', ...)调用参数")
    return r


class DumbFuncGenerator:

    def __init__(self, name):
        self.name = name
        self.interface_doc = None
        self.call_code = None
        self.api_name = None
        self.arg_list = None
        self.ai: BaseAI = None
        self.prompt_user_for_overwrite = CONFIG.get("ask", True)  # 询问是否覆盖
        self.analysis_data = ""
        self.call_idx = 0  # 表示该项目采用的调用部分代码的索引值
        self.dumb_analysis_temprature = CONFIG.get("dumb_analysis_temprature", 0.3)
        self.dumb_generate_code_temprature = CONFIG.get(
            "dumb_generate_code_temprature", 0.3
        )

    def generate_dumb_simulator_function(self, api_name: str, call_idx: int):
        if not ProjectRegistrationInfoReader().can_generate_dumb_simulator(self.name):
            print(
                f"[INS_ERROR] Project {self.name} can not generate dumb simulator, check whether the interface info, interface doc, and adapter all exist"
            )
            return

        # 关键入口
        self.call_idx = call_idx
        start_time = time.time()
        self.api_name = api_name
        self.write_path = (
            DUMB_SIMULATOR_PATH
            + f"{self.name}/"
            + f"call{call_idx}_{self.api_name}_dumbsimulator.py"
        )
        self.analysis_data_path = (
            DUMB_SIMULATOR_PATH
            + f"{self.name}/"
            + f"call{call_idx}_{self.api_name}_dumbsimulator_analysis.md"
        )
        if not os.path.exists(os.path.dirname(self.write_path)):
            os.makedirs(os.path.dirname(self.write_path))
        if (
            os.path.exists(self.write_path)
            and self.prompt_user_for_overwrite
        ):
            print(
                f"[INS_WARN] Non-intelligent module simulation artifact {self.api_name}_call{call_idx}_dumb.py already exists"
            )
            ask = input(f"[INS_WARN] Overwrite? y/n：")
            if ask != "y":
                return
        #  临时逻辑
        # if os.path.exists(self.write_path):
        #     print(
        #         f"[INS_WARN] Non-intelligent module simulation artifact {self.api_name}_call{call_idx}_dumb.py already exists, overwriting"
        #     )
        #     return
        #  临时逻辑结束
        print(
            "[INS_INFO] Analysis process can be viewed anytime at "
            + self.analysis_data_path
        )
        self.call_code = self.get_call_code(call_idx, api_name)
        if not self.call_code:
            raise ValueError(
                f"Cannot find call code for API '{api_name}' with call index {call_idx}"
            )
        origin_script_path = (
            InterfaceInfoReader(self.name)
            .get_call_dict_by_idx(call_idx)
            .get("Path", "")
        )
        future_import_statements, import_statements = extract_import_statements(
            self.call_code
        )  # 原调用代码的导包部分，将会被添加到模拟函数头部
        self.interface_doc = InterfaceDocReader(self.name).get_doc_with_api(
            api_name, self.ai
        )
        history = (
            f"Here is the documentation for the {self.api_name} interface:\n{self.interface_doc}\n\n"
            + f"Here is the code section where the {self.api_name} interface is called:\n{self.call_code}\n\n"
        )
        self.ai.add_history("user", history)
        # 以上处理完毕接口文档，调用代码，API名称，并添加至记忆
        self.arg_list = self.extract_parameters()
        print(f"[INS_INFO] Extracted parameter list: {self.arg_list}")
        get_arg_code_list = self.analyze_parameter(self.arg_list)
        # 以上处理完毕参数分析
        # 生成最终的dumb_simulator
        dumb_simulator_code = self.compose_final(get_arg_code_list)
        full_import_statements = future_import_statements + "\n" + import_statements
        from_import_objects = extract_from_import_object(full_import_statements)
        for obj in from_import_objects:
            dumb_simulator_code = remove_definitions_by_names(
                dumb_simulator_code, obj
            )

        importlist = full_import_statements.splitlines()
        for line in importlist:
            if line.strip() == "":
                continue
            dumb_simulator_code = remove_imports_from_code(
                dumb_simulator_code, imports_to_remove=line
            )

        dumb_simulator_code += "\n\n"
        dead_code_front = f"""
{future_import_statements}
{import_statements}
import sys
sys.path.append('{os.path.dirname(origin_script_path)}')
from Inspection.adapters.custom_adapters.{self.name} import * 
exe = Executor('{self.name}', 'dumb')
exe.set_record_function([''])
OUTPUTS = exe.now_record_path
def set_exe(_exe):
    global exe
    global OUTPUTS
    exe = _exe
    OUTPUTS = exe.now_record_path
# the code top is auto generated, please do not modify it
"""
        dead_code_end = f"""
if __name__ == "__main__":
    print(dumb_simulator())
"""

        dumb_simulator_all_code = dead_code_front + dumb_simulator_code + dead_code_end
        print(
            f"[INS_INFO] Non-intelligent module simulation artifact generated successfully"
        )
        spent_time = time.time() - start_time
        self.analysis_data += (
            f"\n\n\n$$$$$生成dumb_simulator函数耗时: {spent_time}秒$$$$$\n"
        )
        self._write_analysis_data()
        self.write_code_to_file(dumb_simulator_all_code)
        return dumb_simulator_all_code

    def set_base_ai(self, base_ai: BaseAI):
        self.ai = base_ai

    def _write_analysis_data(self):
        with open(self.analysis_data_path, "w") as f:
            f.write(self.analysis_data)
        

    def get_call_code(self, call_idx, api_name):
        dir_name = SIMULATOR_PATH + f"{self.name}/"
        pyfiles = os.listdir(dir_name)
        pyfiles = [f for f in pyfiles if f.endswith(".py")]
        for file in pyfiles:
            match = re.match(r"call(\d+)_(.+)_simulator\.py", file)
            if match:
                idx = int(match.group(1))
                apiname_in_file = match.group(2)
                if idx == call_idx and apiname_in_file == api_name:
                    with open(dir_name + file, "r") as f:
                        code = f.read()
                    return code
        return None

    def extract_parameters(self):
        param_list = []
        try:
            # 尝试从调用代码中提取参数
            param_list = extract_arg_names_for_apiname(self.call_code, self.api_name)
            return param_list
        except Exception as e:
            print(f"[INS_WARN] Failed to extract parameters from call code: {e}")
            # If failed, use AI to extract parameters
            print("[INS_INFO] Using AI to extract parameters")
        ai1 = self.ai.copy(with_memory=True)
        ai2 = self.ai.copy(with_memory=True)
        prompt = f"""
For the above calling code, what parameters are used by the exe.run('{self.api_name}', ...) interface?
Please extract the parameter names used in the calling code. Note that what should be returned are the parameter names defined in the interface documentation, not the variable names input in the calling code.
Return format should be a python list, for example:
['param1', 'param2', 'param3']
Please only return a python list, do not add any other content!!
"""
        response1 = ai1.generate_text(prompt, temperature=self.dumb_analysis_temprature)
        response2 = ai2.generate_text(prompt, temperature=self.dumb_analysis_temprature)
        response = ""
        if response1 != response2:
            ai3 = self.ai.copy(with_memory=True)
            print(
                f"[INS_WARN] Two AI-extracted parameters are inconsistent, starting third extraction"
            )
            prompt = f"""
For the above calling code, what parameters are used by the exe.run('{self.api_name}', ...) interface?
Now two AIs have extracted inconsistent parameters
The first AI extracted parameters: {response1}
The second AI extracted parameters: {response2}
You need to determine which AI extracted the correct parameters. Note that what should be returned are the parameter names defined in the interface documentation, not the variable names input in the calling code.
Please extract the parameter names used in the calling code. Return format should be a python list, for example:
['param1', 'param2', 'param3']
Please only return a python list, do not add any other content!!
"""
            response = ai3.generate_text(
                prompt, temperature=self.dumb_analysis_temprature
            )
        else:
            response = response1

        import ast

        param_list = ast.literal_eval(response)
        if not isinstance(param_list, list):
            raise ValueError(f"参数提取失败，返回值不是列表：{response}")
        self.analysis_data += f"\n\n\n$$$$$参数提取结果$$$$$\n{response}\n"
        self._write_analysis_data()
        return param_list

    def analyze_parameter(self, param_name_list):
        """分析单个参数，追踪来源（多线程实现）"""

        def analyze_single_param(param_name):
            sub_ai = self.ai.copy(with_memory=True)
            print(f"[INS_INFO] Starting analysis of parameter {param_name}")
            analyzer = ParameterAnalyzer(
                sub_ai, self.api_name, param_name, self.call_code, parent=self
            )
            return analyzer.analyze()

        result_list = []
        with ThreadPoolExecutor() as executor:
            futures = {}
            for param_name in param_name_list:
                # 每次提交任务后暂停1秒
                future = executor.submit(analyze_single_param, param_name)
                time.sleep(1)
                futures[future] = param_name
            for future in futures:
                try:
                    result_list.append(future.result())
                except Exception as e:
                    print(
                        f"[INS_ERROR] Parameter {futures[future]} analysis failed: {e}"
                    )
        return result_list

    def wrap_func_bodies(self, func_bodies: list):
        processed_bodies = []
        for func_body in func_bodies:
            first_line = func_body.split("\n")[0]
            # 去除开头的#
            if not first_line.startswith("#"):
                raise ValueError(f"函数体的第一行不是注释：{first_line}")
            first_line = str(first_line[1:])
            # 将所有的空格，换行符号去掉
            first_line = first_line.replace(" ", "")
            param_name = first_line.replace("\n", "")
            # 将func_body中的get_{param_name}()替换为get_{param_name}_inner()
            func_body = func_body.replace(
                f"get_{param_name}()", f"get_{param_name}_inner()"
            )
            func_body = remove_function_calls(func_body, [f"get_{param_name}_inner"])
            func_body += f"\n\n"
            func_body += f"param_{param_name}_value = get_{param_name}_inner()\n"
            func_body += f"return param_{param_name}_value\n"
            # 将fucn_body中的每一行增加一级缩进
            func_body = func_body.replace("\n", "\n    ")
            func_body = f"def get_{param_name}():\n" + "    " + func_body
            processed_bodies.append(func_body)
        return processed_bodies

    def compose_final(self, func_bodies):
        """最终拼接生成dumb_simulator"""
        print(f"[INS_INFO] Starting to splice and merge functions")
        weaped_func_bodies = self.wrap_func_bodies(func_bodies)
        func_bodies_str = "\n\n".join(weaped_func_bodies)
        prompt = f"""
{func_bodies_str}
Please process the above code with the following requirements:
1. Remove example calling code, directly remove import * or from ... import * statements
2. Fix simple syntax errors in the code, such as indentation errors, missing colons
3. You cannot change any semantic logic of the code, even if it seems unreasonable
Generate the final code marked with ```python```
"""
        ai = self.ai.copy()
        response = ai.generate_text(
            prompt, temperature=self.dumb_generate_code_temprature
        )
        self.analysis_data += f"\n\n\n$$$$$代码拼接结果$$$$$\n{response}\n"
        self._write_analysis_data()
        merged_code = extract_python_code(response)
        going_remove_list = ["exe", "ENV_DIR", "OUTPUTS", "RESOURCES_PATH"]
        for going_remove in going_remove_list:
            merged_code = remove_assignments(name=going_remove, code=merged_code)
        dumb_simulator_code = merged_code
        dumb_simulator_code += "\n\ndef dumb_simulator():\n"
        dumb_simulator_code += "    result = {}\n"
        for arg in self.arg_list:
            dumb_simulator_code += f"    try:\n"
            dumb_simulator_code += f"        arg = get_{arg}()\n"
            dumb_simulator_code += f"    except Exception:\n"
            dumb_simulator_code += f"        arg = None\n"
            dumb_simulator_code += f"    result['{arg}'] = arg\n"
        dumb_simulator_code += "    return result\n"
        return dumb_simulator_code

    def write_code_to_file(self, code: str):
        with open(self.write_path, "w") as f:
            f.write(code)
        print(
            f"[INS_INFO] Simulation function generated successfully, saved to: {self.write_path}"
        )
        inj = Injector(self.name)
        injected_code = inj.inject(self.call_idx, self.api_name, self.write_path)
        injected_code_path = self.write_path[:-3] + "_injected.py"
        with open(injected_code_path, "w") as f:
            f.write(injected_code)
        print(
            f"[INS_INFO] Injection code generated successfully, saved to: {injected_code_path}"
        )


class ParameterAnalyzer:
    
    def __init__(self, base_ai, api_name, param_name, call_code, parent):
        self.ai: BaseAI = base_ai
        self.param_name = param_name
        self.api_name = api_name
        self.call_code = call_code
        self.parent: DumbFuncGenerator = parent

    def analyze(self):
        self.ai.add_history(
            "system",
            f"You are a Python static code analysis assistant. \nYour task is to trace the data flow for a specific keyword argument in a call to `exe.run` in the given Python source code.",
        )

        init_prompt = f"""
Focus specifically on the parameter `{self.param_name}` used in `exe.run("{self.api_name}", ...)`.
1. Analyze and explain the complete data flow of `{self.param_name}` throughout the source code:
   - Where it originates;
   - How its value is calculated, retrieved, or derived through function calls;
   - How it is passed, transformed, or modified step by step;
   - How it ultimately reaches `exe.run`.
   - If there are any IF branches or conditional statements affecting the value of `{self.param_name}`, please analyze each branch separately and explain how the value changes in different conditions.
   *This is the main focus — provide as detailed an analysis as possible.*
   **Important Note:**  
   If `{self.api_name}` is invoked inside a loop and `{self.param_name}` changes across iterations, focus strictly on the value from the first iteration.  
   Ignore values from subsequent iterations entirely.
2. Identify the key functions, constants, classes, and any other components involved in obtaining `{self.param_name}`:
   - Specify whether each is defined in the same module or imported from external modules.
The process of obtaining the parameter value is the top priority. Focus on clarity and precision.
"""
        analysis_result = self.ai.generate_text(
            init_prompt, temperature=self.parent.dumb_analysis_temprature
        )
        self.parent.analysis_data += (
            f"$$$$$参数{self.param_name}路径分析$$$$$\n{analysis_result}\n"
        )
        self.parent._write_analysis_data()
        # 分析结束
        print(
            f"[INS_INFO] {self.api_name}'s {self.param_name} analysis completed, starting function generation"
        )
        final_prompt = f"""
After completing the above analysis, please generate a Python function named `get_{self.param_name}` that obtains the value of the `{self.param_name}` parameter for `{self.api_name}`.
**Instructions for the generated function:**
- The function must be named exactly `get_{self.param_name}`.
- It must take no arguments.
- It must fully reproduce the parameter value calculation, using the same methods as observed in the source code, step by step.
- If auxiliary functions or classes are involved and defined in the source code, you must include their full implementation in the generated code.
- You may directly use any modules, functions, classes, and constants that are imported in the source code. Assume all such imports are available at runtime; do not add fallbacks, stubs, or alternative implementations.
- If `{self.api_name}` is invoked inside a loop, ensure that `get_{self.param_name}` only returns the value from the **first iteration** of that loop.
- Please preserve as much of the original logic and code structure as possible when generating the function, including variable names, function calls, and calculation steps. Avoid unnecessary abstraction or simplification.
Please format the generated code block using Markdown triple backticks with `python` as the language identifier.
"""
        final_reply = self.ai.generate_text(
            final_prompt, temperature=self.parent.dumb_generate_code_temprature
        )
        self.parent.analysis_data += (
            f"\n\n\n$$$$$生成的get_{self.param_name}函数代码$$$$$\n{final_reply}\n"
        )
        self.parent._write_analysis_data()
        final_code = extract_python_code(final_reply)
        final_code = f"# {self.param_name}\n" + final_code
        return final_code


if __name__ == "__main__":
    d = DumbFuncGenerator("CodeFormer")
    d.set_base_ai(BaseAI())
