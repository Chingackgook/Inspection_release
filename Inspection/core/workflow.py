from typing import List
from enum import Enum
import os
import sys
import time

from Inspection import (
    RECORD_PATH,
    SIMULATOR_PATH,
    DUMB_SIMULATOR_PATH,
)
from Inspection.utils.config import CONFIG
from Inspection.utils.tools import run_as_module


class StepType(Enum):
    GEN_DOC = "generate_documentation"
    GEN_ADAPTER = "generate_adapter_code"
    GEN_SIMULATION = "generate_simulation_code"
    GEN_DUMB_SIMULATOR = "generate_dumb_simulator_code"
    EXEC_SIMULATION = "execute_simulation_code"
    EXEC_DUMB = "execute_dumb_simulator_code"


class Step:
    def __init__(self, step_name: StepType, **kwargs):
        self.step_name = step_name
        self.params = kwargs
        self.BaseAI = None
        self.sys_path = [p for p in sys.path]
        self.cwd = os.getcwd()

    def set_base_ai(self, base_ai):
        self.BaseAI = base_ai

    def execute(self):
        if self.step_name == StepType.GEN_DOC:
            self._execute_generate_doc()
        elif self.step_name == StepType.GEN_ADAPTER:
            self._execute_generate_adapter()
        elif self.step_name == StepType.GEN_SIMULATION:
            self._execute_generate_simulation()
        elif self.step_name == StepType.GEN_DUMB_SIMULATOR:
            self._execute_generate_dumb_simulator()
        elif self.step_name == StepType.EXEC_SIMULATION:
            self._execute_script(StepType.EXEC_SIMULATION)
        elif self.step_name == StepType.EXEC_DUMB:
            self._execute_script(StepType.EXEC_DUMB)
        else:
            print(f"[INS_ERR] Unknown step: {self.step_name}")
            raise Exception(f"Unknown step: {self.step_name}")

    def _recover_sys_path(self):
        # Since sys.path and os.cwd() are modified in adapter during script execution, need to restore original sys.path after completion
        sys.path = self.sys_path
        if os.getcwd() != self.cwd:
            try:
                os.chdir(self.cwd)
            except Exception as e:
                print(f"[INS_ERR] Failed to restore working directory: {e}")

    def _execute_generate_doc(self):
        from Inspection.generator.doc_generator import DocGenerator

        doc_generator = DocGenerator(self.params["project_name"])
        doc_generator.set_base_ai(self.BaseAI)
        doc_generator.generate_doc()

    def _execute_generate_adapter(self):
        from Inspection.generator.adapter_generator import AdapterGenerator

        adapter_generator = AdapterGenerator(self.params["project_name"])
        adapter_generator.set_base_ai(self.BaseAI)
        adapter_generator.generate_adapter()

    def _execute_generate_simulation(self):
        from Inspection.generator.simulation_generator import SimulationGenerator

        simulation_generator = SimulationGenerator(self.params["project_name"])
        simulation_generator.set_base_ai(self.BaseAI)
        simulation_generator.generate_simulation(
            api_name=self.params["api_name"], call_idx=self.params["call_idx"]
        )

    def _execute_generate_dumb_simulator(self):
        from Inspection.generator.dumb_func_generator import DumbFuncGenerator

        dumb_func_generator = DumbFuncGenerator(self.params["project_name"])
        dumb_func_generator.set_base_ai(self.BaseAI)
        dumb_func_generator.generate_dumb_simulator_function(
            api_name=self.params["api_name"], call_idx=self.params["call_idx"]
        )

    def _execute_script(self, interface_type):
        exec_file_path = self._get_script_file_path(interface_type)
        if exec_file_path:
            syntaxFail = False
            try:
                with open(exec_file_path, "r") as f:
                    code = f.read()
                    code_record_folder = os.path.join(
                        RECORD_PATH, "code_records", self.params["project_name"]
                    )
                    if not os.path.exists(code_record_folder):
                        os.makedirs(code_record_folder)
                    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    code_record_path = os.path.join(
                        code_record_folder,
                        f"{self.params['project_name']}_{interface_type}_{now}.txt",
                    )
                    with open(code_record_path, "w") as code_f:
                        code_f.write(code)
                    auto_feed_enter = CONFIG.get("exec_auto_feed_enter", False)
                    if CONFIG.get("exec_use_subprocess", False):
                        run_as_module(exec_file_path, feed_enter=auto_feed_enter)
                    else:
                        if auto_feed_enter:
                            raise NotImplementedError(
                                "Auto feed enter is only supported in subprocess execution mode."
                            )
                        script_globals = globals().copy()
                        script_globals["__name__"] = "__main__"
                        exec(code, script_globals)
            except SyntaxError as e:
                print(f"[INS_ERR] {interface_type} code syntax error: {e}")
                syntaxFail = True
                raise e
            except IndentationError as e:
                print(f"[INS_ERR] {interface_type} code syntax error: {e}")
                syntaxFail = True
                raise e
            # 建议生成部分
            gen_suggestion = CONFIG.get("auto_suggest", False)
            if not syntaxFail and gen_suggestion:
                from Inspection.generator.suggestion_generator import (
                    SuggestionGenerator,
                )

                suggestion_generator = SuggestionGenerator(self.params["project_name"])
                suggestion_generator.set_base_ai(self.BaseAI)
                simulation_type = ""
                if interface_type == StepType.EXEC_SIMULATION:
                    simulation_type = "simulation"
                elif interface_type == StepType.EXEC_DUMB:
                    simulation_type = "dumb"
                suggestion_generator.generate_suggestions(
                    api_name=self.params["api_name"],
                    idx=self.params["call_idx"],
                    simulate_type=simulation_type,
                )
            self._recover_sys_path()
        else:
            print(f"[INS_ERR] {interface_type} code file path not found.")
            raise Exception(f"{interface_type} code file path not found.")

    def _get_script_file_path(self, interface_type: StepType):
        exec_file_path = None
        if interface_type == StepType.EXEC_SIMULATION:
            simulateidx = self.params["call_idx"]
            api_name = self.params["api_name"]
            # 需要修改，读取文件逻辑，存储文件
            name = f"call{simulateidx}_{api_name}_simulator.py"
            path = os.path.join(SIMULATOR_PATH, self.params["project_name"])
            full_path = os.path.join(path, name)
            if not os.path.exists(full_path):
                print(f"[INS_ERR] {interface_type} code not found: {full_path}")
                raise Exception(f"{interface_type} code not found: {full_path}")
            exec_file_path = full_path

        elif interface_type == StepType.EXEC_DUMB:
            path = os.path.join(DUMB_SIMULATOR_PATH, self.params["project_name"])
            api_name = self.params["api_name"]
            call_idx = self.params["call_idx"]
            py_files = os.listdir(path)
            exec_file_path = None
            for pyfile in py_files:
                name = f"call{call_idx}_{api_name}_dumbsimulator_injected.py"
                if pyfile == name:
                    exec_file_path = os.path.join(path, pyfile)
                    break
            if not exec_file_path:
                print(f"[INS_ERR] {interface_type} code not found: {path}")
                raise Exception(f"{interface_type} code not found: {path}")
        else:
            print(f"[INS_ERR] Unknown interface type: {interface_type}")
            raise Exception(f"Unknown interface type: {interface_type}")
        return exec_file_path


class Workflow:
    def __init__(self, simulate_pj_name: str, sys_additional_path: List, base_ai):
        self.base_ai = base_ai
        self.now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        self.base_ai.id = f"AIRecord_{simulate_pj_name}_{self.now}"
        self.simulate_pj_name = simulate_pj_name
        for p in sys_additional_path:
            if p not in sys.path:
                sys.path.append(p)
        self.steps: List[Step] = []
        self.current_step = 0

    def write_error_log(self, error_message):
        err_path_base = RECORD_PATH + "error/"
        if not os.path.exists(err_path_base):
            os.makedirs(err_path_base)
        err_name = f"err_{self.simulate_pj_name}_{self.now}.txt"
        err_path = os.path.join(err_path_base, err_name)
        with open(err_path, "a") as f:
            f.write(f"[INS_ERR] {error_message}\n")
        print(f"[INS_ERR] {error_message}")
        print(f"[INS_ERR] Error message has been logged to: {err_path}")

    def add_step(self, step_name, simulate_idx=None, api_name=None):
        step_params = {
            "project_name": self.simulate_pj_name,
            "call_idx": simulate_idx,
            "api_name": api_name,
        }
        step = Step(step_name, **step_params)
        step.set_base_ai(self.base_ai.copy())
        self.steps.append(step)

    def run(self):
        for step in self.steps:
            try:
                step.execute()
            except Exception as e:
                import traceback

                print(f"[INS_ERR] Step execution failed: {step.step_name}")
                traceback_info = traceback.format_exc()
                self.write_error_log(
                    f"Step execution failed: {step.step_name}, error message: {e}\nStack trace: \n{traceback_info}"
                )
                raise e
            self.current_step += 1

    def run_step(self, step_name: StepType, call_idx=None, api_name=None):
        try:
            step_params = {
                "project_name": self.simulate_pj_name,
                "call_idx": call_idx,
                "api_name": api_name,
            }
            step = Step(step_name, **step_params)
            step.set_base_ai(self.base_ai.copy())
            step.execute()
        except Exception as e:
            import traceback

            print(f"[INS_ERR] Step execution failed: {step_name.value}")
            traceback_info = traceback.format_exc()
            self.write_error_log(
                f"Step execution failed: {step_name.value}, error message: {e}\nStack trace: \n{traceback_info}"
            )
            raise e

    def clear_steps(self):
        self.steps.clear()
        self.current_step = 0
