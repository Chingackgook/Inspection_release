# Workflow file rules
# Step name + parameters, separated by English commas, start with the project name to be simulated, new line represents new step
# Each step uses a new line
# The first step represents the workflow name
# Example:
"""
CLIP
generate_doc
generate_adapter
generate_simulation, some_api, 1(reference to call, starting from 0)
generate_dumb_simulator, some_api, 1(reference to call, starting from 0)
test
simulation, some_api, 1(reference to call, starting from 0)
dumb, some_api , 1(reference to call, starting from 0)
"""
# The above step name abbreviations are g_doc, g_adp, g_tes, g_sim, g_dum, t, s, d respectively
# Now supports bracket syntax and nesting, steps are separated by English semicolons
# For example: 3(generate_dumb_simulator, some_api, 1 ; 2(dumb, some_api_name , 1))

from Inspection.core.workflow import Workflow, StepType
from Inspection.utils.config import CONFIG
import os

sys_path = CONFIG.get("path", [])


def decode_line(code: str) -> list:
    units = []
    # For the left side of the left bracket, if it's not a number, is a semicolon or string start, add a 1 before it
    k = 0
    while k < len(code):
        if code[k] == "(":
            if k == 0:
                code = "1" + code
                k += 1
                continue
            elif code[k - 1] == ";":
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
            new_code = code[i + 1 : j - 1]
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

    def __compile_line(self, line: str) -> str:
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
            with open(self.code_or_path, "r") as file:
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
        Parse each line of text and add to workflow based on step name.
        Assumes format: step_name, parameter1=value1, parameter2=value2, ...
        """
        if self.lines is None or len(self.lines) == 0:
            print(f"[INS_ERR] File is empty or not read: {self.code_or_path}")
            return
        first_line = self.lines[0].replace(" ", "")
        if first_line[-1] == ";":
            first_line = first_line[:-1]
        self.workflow = Workflow(first_line, sys_path, self.BaseAI)
        self.lines = self.lines[1:]  # Remove the first line
        for line in self.lines:
            parts = line.split(",")
            step_name = parts[0].strip()
            if step_name == "generate_doc" or step_name == "g_doc":
                self.workflow.add_step(StepType.GEN_DOC)
            elif step_name == "generate_adapter" or step_name == "g_adp":
                self.workflow.add_step(StepType.GEN_ADAPTER)
            elif step_name == "generate_simulation" or step_name == "g_sim":
                api_name = parts[1].strip()
                idx = int(parts[2].strip())
                self.workflow.add_step(
                    StepType.GEN_SIMULATION, simulate_idx=idx, api_name=api_name
                )
            elif step_name == "generate_dumb_simulator" or step_name == "g_dum":
                api_name = parts[1].strip()
                idx = int(parts[2].strip())
                self.workflow.add_step(
                    StepType.GEN_DUMB_SIMULATOR, api_name=api_name, simulate_idx=idx
                )
            elif step_name == "simulation" or step_name == "s":
                api_name = parts[1].strip()
                idx = int(parts[2].strip())
                self.workflow.add_step(
                    StepType.EXEC_SIMULATION, simulate_idx=idx, api_name=api_name
                )
            elif step_name == "dumb" or step_name == "d":
                idx = int(parts[2].strip())
                api_name = parts[1].strip()
                self.workflow.add_step(
                    StepType.EXEC_DUMB, simulate_idx=idx, api_name=api_name
                )
            else:
                print(f"[INS_WARN] Unknown step: {step_name}")
                continue
        return self.workflow


# Usage example
# from Inspection.ai.base_ai import BaseAI
# from Inspection import WORKFLOW_PATH
# if __name__ == "__main__":
#     # Assume you have a workflow file 'workflow.txt'
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
