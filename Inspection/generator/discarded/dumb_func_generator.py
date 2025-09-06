from Inspection import SIMULATION_PATH , CUSTOM_ADAPTER_PATH ,ENV_BASE
from Inspection.utils.config import CONFIG
from Inspection.ai.base_ai import BaseAI
from Inspection.utils.interface_reader import InterfaceDocReader, InterfaceInfoReader
from Inspection.core.code_processor import extract_python_code , remove_assignments
from Inspection.core.code_processor import get_class_code
from Inspection.generator.injector import Injector

import os


def remove_exe_assignments(code):
    return remove_assignments('exe', code)

def remove_ENV_DIR_assignments(code):
    result = remove_assignments('ENV_DIR', code)
    return result

def optimize_code(code, ai : BaseAI = None):
    """
    优化代码
    :param code: 需要优化的代码
    :return: 优化后的代码
    """
    promote = f"""
{code}
检查这段代码的简单语法错误，并修复
生成的代码用```python ... ```包裹，只用返回代码
"""
    result = ai.generate_text(promote)
    code = extract_python_code(result)
    return code



class DumbFuncGenerator:
    """
    用于生成非智能化模块的模拟执行函数
    """
    def __init__(self, name):

        self.name = name
        self.BaseAI = None
        self.use_simulation = CONFIG.get('dumb_use_simulation', False)
        self.ask = CONFIG.get('ask', True) # 询问是否覆盖

    def set_base_ai(self, base_ai: BaseAI):
        self.BaseAI = base_ai

    def generate_dumb_simulator_function(self , api_name: str , idx:int = 0):
        """
        api_name: 接口名称（这里为适配器里已经注册到exe.run的函数名称）
        idx: 选取interface_call中第几个作为模拟执行参考
        生成与这个api有关的非智能化模块的模拟执行函数
        """
        path = SIMULATION_PATH + '/dumb_simulator/' + f'{self.name}/' + api_name + f'_call{idx}_dumb.py'
        if os.path.exists(path) and self.ask:
            print(f"[INS_WARN] Non-intelligent module simulation function {api_name}_call{idx}_dumb.py already exists")
            ask = input(f"[INS_WARN] Overwrite? y/n：")
            if ask != 'y':
                return
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        
        interface_doc = InterfaceDocReader(self.name).get_doc()
        call_data = None
        if not self.use_simulation:
            call_data = InterfaceInfoReader(self.name).get_call_str_by_idx(idx)
        else:
            temp_path = SIMULATION_PATH + '/simulate_interface/' + f'{self.name}/'
            # 取出该文件夹内的以 idx 开头的python文件
            files = [f for f in os.listdir(temp_path) if f.startswith(str(idx)) and f.endswith('.py')]
            if len(files) == 0:
                print(f"[INS_WARN] Simulation execution file does not exist")
                return
            call_data = open(temp_path + files[0], 'r').read()
        content = f"""
现有一个接口文档
{interface_doc}
你需要关注的内容是{api_name}的输入参数部分
"""
        self.BaseAI.add_history('user', content)
        print(f'[INS_INFO] Calculating parameter values...')
        promote = f"""
{api_name}的被调用的部分为：
{call_data}
请你读懂代码流程，定位{api_name}的输入参数部分，分析并告诉我每个参数的完整源头路径，
请尽全力详细分析到最源头部分，并注意条件语句
计算出每个参数的值，或是通过调用获得参数的值，源代码是如何逐步获取参数值的。
如果{api_name}被调用了多次，请你只分析第一次调用
"""
        text = self.BaseAI.generate_text(promote)
        promote = f"""
这些参数的计算用到了哪些自定义函数或类？或是直接能够计算出值？
如果发现使用了某个函数或类，有以下两种情况：
1、在源代码中未定义，你可以直接使用该函数或类，不要自拟定类或方法的实现；2、源代码中有定义，你需要将这个函数或类的定义代码也添加在上面
并且请分析源代码的输入文件的来源
"""
        text += "\n\n\n"
        text += self.BaseAI.generate_text(promote)
        readme_path = SIMULATION_PATH + '/dumb_simulator/' + f'{self.name}/' + api_name + f'_call{idx}_dumb.md'
        with open(readme_path, 'w') as f:
            f.write(text)
        promote = f"""
为了将参数计算结果返回给我，请你生成一个名为dumb_simulator的无输入函数!其返回值为调用代码将要给{api_name}传入的参数字典
参数值获取方法：1、返回你刚才计算出的参数值；2、采用与源代码相同的方式逐步获取参数值
特别对于输入文件，请通过分析原调用代码，采取相同的方式来获得文件，而不是自行生成文件路径或文件名
你需要返回dumb_simulator函数的代码和必要的库，用```python标记代码块
注意返回的字典需匹配源代码中输入参数个数，参数名，不新增不遗漏
"""
        print(f'[INS_INFO] Generating non-intelligent module simulation function...')
        result = self.BaseAI.generate_text(promote)
        code = extract_python_code(result)
        code = remove_exe_assignments(code)
        code = remove_ENV_DIR_assignments(code)
        dead_code_front =f"""
# {api_name}
from Inspection.adapters.custom_adapters.{self.name} import * 
from Inspection.adapters.custom_adapters.{self.name} import ENV_DIR
from Inspection.executor import Executor
exe = Executor('{self.name}','dumb') #创建一个Executor对象
exe.set_record_function(["{api_name}"]) #设置记录函数
FILE_RECORD_PATH = exe.now_record_path
# 以上代码为自动生成\n\n
"""
        code = dead_code_front + code
        # 检查是否存在exe_init 
        init_path = ENV_BASE + f'{self.name}/exe_init.py'
        find_init = False
        if os.path.exists(init_path):
            find_init = True
            with open(init_path, 'r') as f:
                exe_init_code = f.read()
                result = f'\n#从{init_path}读取的初始化信息\n'+ exe_init_code + '\n\n#end\n'
                find_init = True
        
        if not find_init:
            self.BaseAI.clear_history()
            adapterdata = get_class_code(CUSTOM_ADAPTER_PATH+self.name+'.py', 'CustomAdapter')
            history = f"""
这里是原调用代码{call_data}
这里是一个适配器类的实现{adapterdata}
你需要理解适配器类的create_interface_objects方法，然后进行代码补全
"""
            self.BaseAI.add_history('user', history)
            promote = f"""
你的任务很简单，因为我已经实例化一个CustomAdapter类的对象exe，然后运行exe.run("{api_name}" , **kwargs)来执行原有的{api_name}函数
但因为我并没有调用exe.create_interface_objects来进行接口对象的初始化，有可能无法正常运行，请你结合原调用代码和适配器类的实现
生成一段python代码，补全exe.create_interface_objects部分，用```python标记代码块
注意：只需要代码块，且运行完成exe.create_interface_objects就可以停止生成了
"""
            result = self.BaseAI.generate_text(promote)
            result = extract_python_code(result)
        code += result
        # 使用了injector注入进simulator，不需要最后的调用代码
#         dead_code_end=f"""\n
# args_dict = dumb_simulator() #调用模拟函数生成参数
# exe.run("{api_name}" , **args_dict) #执行该接口
# """
#         code += dead_code_end
        print(f"[INS_INFO] Optimizing code...")
        code = optimize_code(code, self.BaseAI.copy())
        with open(path, 'w') as f:
            f.write(code)
        print(f"[INS_INFO] Non-intelligent module simulation function generated successfully")
        print(f"[INS_INFO] Simulation function code saved to {path}")
        inj = Injector(self.name)
        inj.inject(idx, api_name)
        


if __name__ == '__main__':
    gen = DumbFuncGenerator('whisper')
    gen.set_base_ai(BaseAI())
    gen.generate_dumb_simulator_function('transcribe', 0)



