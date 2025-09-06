# 模拟执行代码生成器
import os

from Inspection.ai.base_ai import BaseAI
from Inspection.core.code_processor import extract_python_code, extract_import_statements
from Inspection import INTERFACE_DOC_PATH, CUSTOM_ADAPTER_PATH, INTERFACE_INFO_PATH
from Inspection.core.code_processor import get_class_code , remove_assignments
from Inspection.core.code_processor import extract_function_names_from_adapter
from Inspection.utils.interface_reader import InterfaceDocReader, InterfaceInfoReader
from Inspection.utils.path_manager import SIMULATION_PATH
from Inspection.utils.config import CONFIG

def optimize_code(source_code: str, ai_instance: BaseAI = None):
    print('[INS_INFO] Performing final optimization ...')
    optimization_prompt = f"""
{source_code}
任务：
1、检查这段代码的简单语法错误，并修复
2、将这段代码中所有将输出文件的根路径替换为一个已有的全局变量FILE_RECORD_PATH（已经存在，无须定义）
3、将这段代码中的所有将输入文件(包括配置文件，模型路径等)的根路径替换为一个已有的全局变量ENV_DIR（已经存在，无须定义）
4、如果这段代码使用if __name__ == '__main__'运行主逻辑，请去除if __name__ == '__main__'，改为直接运行主逻辑
5、如果出现输入或输出为io.BytesIO的情况，请将其文件路径已有的全局变量FILE_RECORD_PATH+/test/或ENV_DIR+/test/
重要：请最大限度地保留原始代码的结构和逻辑！！，并返回修改后的代码
生成的代码用```python ... ```包裹
""" 
    result = ai_instance.generate_text(optimization_prompt)
    optimized_code = extract_python_code(result)
    from Inspection.core.code_processor import transform_exe_attributes_and_call_name
    optimized_code = transform_exe_attributes_and_call_name(optimized_code,exceptlist=['now_record_path'])
    return optimized_code

def optimize_interface_objects_initialization_code(simulation_code: str, ai_instance: BaseAI = None, original_call_code: str = None, adapter_code: str = None):
    print('[INS_INFO] Optimizing model initialization part ...')
    initialization_prompt = f"""
{adapter_code}
以上是一个适配器类的实现
{original_call_code}
这里是一个原有的调用代码
请理解一下适配器类的create_interface_objects方法与原有调用代码的接口类初始化部分的关系
以及在create_interface_objects方法中初始化了哪些对象进入self中
"""
    _ = ai_instance.generate_text(initialization_prompt)
    simulation_prompt = f"""
{simulation_code}
这里是我的一个模拟执行代码，使用了exe对象模拟原有接口，但是我的代码并没有调用exe.create_interface_objects来进行接口对象初始化
1、请你根据原调用代码的初始化部分，为我的模拟代码补全exe.create_interface_objects的调用。
2、如果create_interface_objects方法中初始化的对象，在模拟代码中未被正确使用，请替换为`exe.对象名`的形式
生成完整的用```python ... ```包裹的代码，其他逻辑保持不变
"""
    # going change: 需要将第二点独立出来，单独优化
    result = ai_instance.generate_text(simulation_prompt)
    optimized_code = extract_python_code(result)
    return optimized_code

class SimulationGenerator:
    def __init__(self, name: str):
        self.name = name
        self.base_ai_instance = None
        self.prompt_user_for_overwrite = CONFIG.get('ask', True)  # 询问是否覆盖
    
    def set_base_ai(self, base_ai_instance: BaseAI):
        self.base_ai_instance = base_ai_instance

    def generate_test_code(self):
        # 生成根据接口文档的测试代码
        # 已弃用
        test_code_path = SIMULATION_PATH + 'test_interface/' + self.name + '.py'
        if os.path.exists(test_code_path) and self.prompt_user_for_overwrite:
            print(f"[INS_WARN] Test interface code {self.name} already exists")
            user_choice = input("Overwrite? (y/n)")
            if user_choice != 'y':
                return
        try:
            interface_documentation = InterfaceDocReader(self.name).get_doc()
        except FileNotFoundError:
            print(f"[INS_ERR] Test execution code generation failed, please check if {INTERFACE_DOC_PATH} contains interface documentation")
            return
        adapter_code = get_class_code(CUSTOM_ADAPTER_PATH + self.name + '.py', 'CustomAdapter')
        available_functions = extract_function_names_from_adapter(adapter_code)
        available_functions_str = '\n'.join([f" - {func}" for func in available_functions])
        print('[INS_INFO] Generating test plan')
        test_plan_prompt = interface_documentation + """ \n
请根据以上接口文档，给出一个测试思路流程（非代码）
"""
        test_plan = self.base_ai_instance.generate_text(test_plan_prompt)
        with open(SIMULATION_PATH + 'test_interface/' + self.name + '.md', 'w') as file:
            file.write(test_plan)
        print(f"[INS_INFO] Test plan generated successfully, saved to {SIMULATION_PATH}test_interface/{self.name}.md")
        print('[INS_INFO] Generating test interface code...')
        test_code_prompt = f"""
现在的一些关键函数都被一个已实现的exe对象封装，已实现的函数：{available_functions_str}，未实现的不要测试，或者请尽量保持原样
调用函数方法：exe.run("function_name" , **kwargs) ，其中function_name为函数名，kwargs为函数参数，此方法会返回原函数的返回值！
请用python实现刚刚的思路流程，使用try语句包裹exe.run进行测试，生成的代码用```python ... ```包裹
如果需要加载接口对象，请使用exe.create_interface_objects(**kwargs)方法
信息：
请勿重复实现！
"""
        test_code = self.base_ai_instance.generate_text(test_code_prompt)
        test_code = f"""
import numpy as np
import sys
import os
from Inspection.adapters.custom_adapters.{self.name} import ENV_DIR
from Inspection.adapters.custom_adapters.{self.name} import *
from Inspection.executor import Executor
exe = Executor('{self.name}', 'test')
FILE_RECORD_PATH = exe.now_record_path

"""  +   remove_assignments('exe',extract_python_code(test_code),use_regex=True)  # 去除exe赋值
        
        optimized_test_code = optimize_code(test_code, self.base_ai_instance.copy())
        with open(test_code_path, 'w') as file:
            file.write(optimized_test_code)
        print(f"[INS_INFO] Test interface code {self.name} generated successfully")

    def generate_simulation(self, idx :int = 0):
        # 生成根据调用信息，以及原调用步骤生成的模拟执行代码
        simulation_path = SIMULATION_PATH + 'simulate_interface/' + self.name+ '/'
        try:
            info_reader = InterfaceInfoReader(self.name)
        except FileNotFoundError:
            print(f"[INS_ERR] Simulation execution code generation failed, please check if {INTERFACE_INFO_PATH} contains interface information and if json format is correct")
            return
        try:
            doc = InterfaceDocReader(self.name).get_doc()
        except FileNotFoundError:
            print(f"[INS_ERR] Simulation execution code generation failed, please check if {INTERFACE_DOC_PATH} contains interface documentation")
            return
        call_data= info_reader.get_call_str_by_idx(idx)
        call_name = info_reader.get_call_dict_by_idx(idx).get('Name', 'unknown')
        future_import , import_data = extract_import_statements(call_data)
        # 检查文件是否已经存在模拟执行代码
        if os.path.exists(simulation_path + str(idx) + '_' + call_name + '.py') and self.prompt_user_for_overwrite:
            print(f"[INS_WARN] Simulation execution code {self.name}/ {idx}_{call_name} already exists")
            ch = input("Overwrite? (y/n)")
            if ch != 'y':
                return
        # 创建目录
        if not os.path.exists(simulation_path):
            os.makedirs(simulation_path)
        exe_data = get_class_code(CUSTOM_ADAPTER_PATH+self.name+'.py', 'CustomAdapter')
        avaliable_fuc = extract_function_names_from_adapter(exe_data)
        avaliable_fuc_str = '\n'.join([f" - {fuc}" for fuc in avaliable_fuc])

        # 以上是该方法的一些准备工作，下面开始进行ai生成模拟执行代码



        print(f'[INS_INFO] Generating simulation execution plan id={idx}')
        promote_replace_exe = f"""
这里是接口文档{doc}
这里是调用部分源代码{call_data}
这里关键的函数或方法为：{avaliable_fuc_str}
现在刚刚提到的关键函数都被一个已实现的exe对象封装，需要将刚刚提到的函数替换为exe.run("function_name" , **kwargs)的形式
你需要了解源代码是如何调用智能化模块的关键函数，并替换为exe.run("function_name" , **kwargs)的形式（exe对象已经被我实现，不需要分析其实现）
exe.run("function_name" , **kwargs) 的返回值是原函数的返回值
生成一个方案（不需要代码）
"""
        # doc为接口文档，call_data为调用部分源代码，avaliable_fuc为注册在适配器中的函数
        way1 = self.base_ai_instance.generate_text(promote_replace_exe)
        promote_add_super_param = f"""
我希望这个代码是能够使用python的exec函数直接运行的，所以你可能需要为原代码添加一些模拟的参数，用于模拟用户输入或是运行时行为
请先分析这段代码的逻辑，然后帮我分析如何在源代码尽量不变的情况下，能够让这个代码被无参数exec执行
生成的代码不能有argparse或是input等交互式的输入，也不能使用@click等命令行参数解析。
生成一个方案（不需要代码，不要模拟exe对象）
"""
        way2 = self.base_ai_instance.generate_text(promote_add_super_param)
        with open(simulation_path + str(idx) + '_' + call_name + '.md', 'w') as f:
            f.write(way1 + '\n' + way2)
        print(f"[INS_INFO] Simulation execution plan generated successfully, saved to {SIMULATION_PATH}simulate_interface/{self.name}/{idx}_{call_name}.md")
        print('[INS_INFO] Generating simulation execution code...')
        genpath = simulation_path + str(idx) + '_' + call_name + '.py'
        self.base_ai_instance.clear_history()
        promote = f"""
源代码：
{call_data}\n
方案1：
{way1}\n
方案2：
{way2}\n
目标：按照上面的两个方案，替换源代码中的方法或函数，为exe.run()，并让代码能直接exec运行，生成的代码用```python ... ```包裹
只有\n{avaliable_fuc_str}\n函数是可用exe.run替换的，其他函数请保持原样!请尽量保留原有的代码结构和逻辑!
注意：我希望能够模拟各个接口的运行状况，所以我将直接无参数运行你生成的代码，并且exe对象已经被我实现
"""
        result = self.base_ai_instance.generate_text(promote)
        result = extract_python_code(result) # 提取python代码
        result = remove_assignments('exe', result, use_regex=True)  # 去除exe赋值
        result = f"""
{future_import}
import numpy as np
import sys
import os
from Inspection.executor import Executor
from Inspection.adapters.custom_adapters.{self.name} import ENV_DIR
from Inspection.adapters.custom_adapters.{self.name} import *
exe = Executor('{self.name}','simulation')
FILE_RECORD_PATH = exe.now_record_path
# 导入原有的包
{import_data}
"""  +  result
        optimized_result = optimize_interface_objects_initialization_code(result, self.base_ai_instance.copy(), call_data, exe_data)
        optimized_result = optimize_code(optimized_result, self.base_ai_instance.copy())
        with open(genpath, 'w') as f:
            f.write(optimized_result)
        print(f"[INS_INFO] Simulation execution code {self.name} generated successfully, saved to {genpath}")

    def genrate_all_simulation(self):
        # Generate all simulation execution code
        try:
            info_reader = InterfaceInfoReader(self.name)
        except FileNotFoundError:
            print(f"[INS_ERR] Simulation execution code generation failed, please check if {INTERFACE_INFO_PATH} contains interface information and if json format is correct")
            return
        for i in range(len(info_reader.info_json['API_Calls'])):
            self.generate_simulation(i)
            self.base_ai_instance.clear_history()
        

if __name__ == "__main__":
    simulation=SimulationGenerator('bark')
    simulation.set_base_ai(BaseAI())
    simulation.generate_simulation(0)
    # print(get_class_code(BASE_DIR+'/Inspection/adapters/custom_adapters/clip.py', 'CustomAdapter'))
