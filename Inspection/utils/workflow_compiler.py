# 工作流文件规则
# 步骤名 + 参数 ，用英文逗号分隔 ，开头为需要模拟执行的项目名，新起一行代表新步骤
# 每个步骤使用新行表示
# 第一个步骤代表工作流名称
# 示例：
"""
CLIP
generate_doc
generate_adapter
generate_test
generate_simulation, 1(引用的call,0开始)
generate_dumb_simulator, some_api, 1(引用的call,0开始)
test
simulation, 1(将要模拟执行的id,0开始)
dumb, some_api_name , 1(引用的call,0开始)
"""
# 以上步骤名简写分别为g_doc, g_adp, g_tes, g_sim, g_dum, t, s, d
# 现支持括号语法以及嵌套，步骤用英文分号分隔
# 例如 3(generate_dumb_simulator, some_api, 1 ; 2(dumb, some_api_name , 1))

from Inspection.core.workflow import Workflow
from Inspection.utils.config import CONFIG
import os
sys_path = CONFIG.get('path', [])

def decode_line(code : str) -> list:
    units = []
    # 对于左括号左边，如果不是数字，是分号或是字符串开头，则在此之前添加一个1
    k = 0
    while k < len(code):
        if code[k] == "(":
            if k == 0:
                code = "1" + code
                k += 1
                continue
            elif code[k-1] == ";":
                code = code[:k] + "1" + code[k:]
                k += 1
                continue
        k += 1
    i = 0
    while i < len(code):
        if code[i].isdigit():
            j = i
            while j < len(code) and code[j].isdigit():
                j += 1
            mult = int(code[i:j])
            i = j
            j += 1
            barket = 1
            while barket > 0 and j < len(code):
                if code[j] == "(":
                    barket += 1
                elif code[j] == ")":
                    barket -= 1
                j += 1
            new_code = code[i+1:j-1]
            new_units = decode_line(new_code)
            for k in range(mult):
                units.extend(new_units)
            i = j
            if i == len(code):
                break
        else:
            j = i
            while j < len(code) and code[j] != ";":
                j += 1
            new_str = code[i:j]
            if new_str != "":
                units.append(code[i:j])
            i = j
        i += 1
    return units



class WorkflowCompiler:
    def __init__(self, code_or_path, base_ai):
        self.code_or_path = code_or_path
        self.BaseAI = base_ai
        self.lines: list[str] = []
        self.workflow = None
        self.__preprocess()

    def __compile_line(self, line:str) -> str:
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        line = line.split("#")[0]
        if line == "":
            return
        units = decode_line(line)
        for unit in units:
            self.lines.append(unit)
        return

    def __preprocess(self):
        if os.path.isfile(self.code_or_path):
            with open(self.code_or_path, 'r') as file:
                lines = file.readlines()
            for line in lines:
                self.__compile_line(line)
        else:
            lines = self.code_or_path.split("\n")
            for line in lines:
                self.__compile_line(line)
        if len(self.lines) == 0:
            print(f"[INS_ERR] File is empty or not read: {self.code_or_path}")

    def compile(self):
        """
        解析每行文本，并根据步骤名称添加到工作流。
        假设格式为: 步骤名称, 参数1=值1, 参数2=值2, ...
        """
        if self.lines is None or len(self.lines) == 0:
            print(f"[INS_ERR] File is empty or not read: {self.code_or_path}")
            return
        first_line = self.lines[0].replace(" ", "")
        if first_line[-1] == ";":
            first_line = first_line[:-1]
        self.workflow = Workflow(first_line ,sys_path , self.BaseAI)
        self.lines = self.lines[1:] # 去掉第一行
        for line in self.lines:
            parts = line.split(",")
            step_name = parts[0].strip()
            if step_name == "generate_doc" or step_name == "g_doc":
                self.workflow.add_step("generate_doc")
            elif step_name == "generate_adapter" or step_name == "g_adp":
                self.workflow.add_step("generate_adapter")
            elif step_name == "generate_test" or step_name == "g_tes":
                self.workflow.add_step("generate_test")
            elif step_name == "generate_simulation" or step_name == "g_sim":
                idx = int(parts[1].strip())
                self.workflow.add_step("generate_simulation", simulate_idx=idx)
            elif step_name == "generate_dumb_simulator" or step_name == "g_dum":
                api_name = parts[1].strip()
                idx = int(parts[2].strip())
                self.workflow.add_step("generate_dumb_simulator", api_name=api_name, simulate_idx=idx)
            elif step_name == "test" or step_name == "t":
                self.workflow.add_step("test")
            elif step_name == "simulation" or step_name == "s":
                idx = int(parts[1].strip())
                self.workflow.add_step("simulation", simulate_idx=idx)
            elif step_name == "dumb" or step_name == "d":
                idx = int(parts[2].strip())
                api_name = parts[1].strip()
                self.workflow.add_step("dumb", simulate_idx=idx, api_name=api_name)
            else :
                print(f"[INS_WARN] Unknown step: {step_name}")
                continue
        return self.workflow

# 用法示例
# from Inspection.ai.base_ai import BaseAI
# from Inspection import WORKFLOW_PATH
# if __name__ == "__main__":
#     # 假设你有一个工作流文件 'workflow.txt'
#     file_path = WORKFLOW_PATH + 'test.txt'
#     base_ai = BaseAI()
#     compiler = WorkflowCompiler(file_path, base_ai)
#     workflow = compiler.compile()
#     if workflow:
#         workflow.run()

if __name__ == "__main__":
    ls = """
CLIP
6(generate_doc ; 3(FDSDSFSDFSDFDSFDSS;)) ; DUM; DDD
generate_adapter;
generate_test
generate_simulation , 1;
generate_dumb_simulator, some_api, 1
test
simulation, 1
3(dumb, some_api_name , 1 ; 2(generate_dumb_simulator, some_api, 1; 2(dumb, some_api_name , 1)))
"""
    compiler = WorkflowCompiler(ls, None)
    for line in compiler.lines:
        print(line)