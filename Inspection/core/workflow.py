from typing import List
import os
import sys
import time


from Inspection import CUSTOM_ADAPTER_PATH , RECORD_PATH ,INSPECTION_DIR , BASE_DIR, EVALUATE_RQ1_PATH
from Inspection.utils.config import CONFIG
from Inspection.core.code_processor import get_class_code



def run_as_module(file_path: str):
    import subprocess
    from pathlib import Path
    """
    :param file_path: Python file path to execute
    :param project_root: Project root directory (optional), if not specified, search upward from file directory
    """
    file_path = Path(file_path).resolve()
    project_root = Path(BASE_DIR).resolve()
    try:
        rel_path = file_path.relative_to(project_root)
    except ValueError:
        raise ValueError(f"File {file_path} is not under project root {project_root}")
    # Remove .py suffix
    if file_path.suffix != ".py":
        raise ValueError(f"{file_path} is not a .py file")
    rel_path = rel_path.with_suffix('')
    # Path separator -> dot
    module_name = '.'.join(rel_path.parts)
    subprocess.run(["python", "-m", module_name], check=True)




class Step:
    def __init__(self, step_name, **kwargs):
        self.step_name = step_name
        self.params = kwargs
        self.BaseAI = None
        self.sys_path = [p for p in sys.path]
        self.cwd = os.getcwd()

    def set_base_ai(self, base_ai):
        self.BaseAI = base_ai

    def execute(self):
        step_map = {
            'generate_doc': self.__execute_generate_doc,
            'generate_adapter': self.__execute_generate_adapter,
            'generate_test': self.__execute_generate_test,
            'generate_simulation': self.__execute_generate_simulation,
            'generate_dumb_simulator': self.__execute_generate_dumb_simulator,
            'test': self.__execute_test,
            'simulation': self.__execute_simulation,
            'dumb': self.__execute_dumb,
        }
        print(f"[INS_INFO] Executing step: {self.step_name}")
        step_map.get(self.step_name, self.__unknown_step)()
    
    def __recover_sys_path(self):
        # Since sys.path and os.cwd() are modified in adapter during script execution, need to restore original sys.path after completion
        sys.path = self.sys_path
        if os.getcwd() != self.cwd:
            try:
                os.chdir(self.cwd)
            except Exception as e:
                print(f"[INS_ERR] Failed to restore working directory: {e}")


    def __unknown_step(self):
        print(f"[INS_ERR] Unknown step: {self.step_name}")

    def __execute_generate_doc(self):
        from Inspection.generator.doc_generator import DocGenerator
        doc_generator = DocGenerator(self.params['name'])
        doc_generator.set_base_ai(self.BaseAI)
        doc_generator.generate_doc()

    def __execute_generate_adapter(self):
        from Inspection.generator.adapter_generator import AdapterGenerator
        adapter_generator = AdapterGenerator(self.params['name'])
        adapter_generator.set_base_ai(self.BaseAI)
        adapter_generator.generate_adapter()
    
    def __execute_generate_test(self):
        from Inspection.generator.discarded.simulation_generator import SimulationGenerator
        simulation_generator = SimulationGenerator(self.params['name'])
        simulation_generator.set_base_ai(self.BaseAI)
        simulation_generator.generate_test_code()

    def __execute_generate_simulation(self):
        if CONFIG.get('simulation_use_v2', True):
            from Inspection.generator.simulation_generator_v2 import SimulationGenerator
        else:
            from Inspection.generator.discarded.simulation_generator import SimulationGenerator
        idx = self.params['idx']
        simulation_generator = SimulationGenerator(self.params['name'])
        simulation_generator.set_base_ai(self.BaseAI)
        if idx is None:
            simulation_generator.genrate_all_simulation()
        else:
            simulation_generator.generate_simulation(call_idx = idx)


    def __execute_generate_dumb_simulator(self):
        use_v2 = CONFIG.get('dumb_use_v2', True)
        if use_v2:
            from Inspection.generator.dumb_func_generator_v2 import DumbFuncGenerator
        else:
            from Inspection.generator.discarded.dumb_func_generator import DumbFuncGenerator
        dumb_func_generator = DumbFuncGenerator(self.params['name'])
        dumb_func_generator.set_base_ai(self.BaseAI)
        dumb_func_generator.generate_dumb_simulator_function(self.params['api_name'], self.params['idx'])

    def __execute_test(self):
        self.__execute_script('test_interface')

    def __execute_simulation(self):
        self.__execute_script('simulate_interface')

    def __execute_dumb(self):
        self.__execute_script('dumb_interface')

    def __execute_script(self, interface_type):
        exec_file = None
        if interface_type == 'test_interface':
            path = os.path.join(os.path.dirname(INSPECTION_DIR), 'simulation', interface_type, f"{self.params['name']}.py")
            if not os.path.exists(path):
                print(f"[INS_ERR] {interface_type} code not found: {path}")
                return
            exec_file = path
        elif interface_type == 'simulate_interface':
            simulateidx = self.params['idx']
            path = os.path.join(os.path.dirname(INSPECTION_DIR),
                                 'simulation', interface_type, self.params['name'])
            use_v2 = CONFIG.get('simulation_use_v2', True)
            if use_v2:
                py_files = [f for f in os.listdir(path) if f.startswith(str(simulateidx)) and f.endswith('_v2.py')]
                if not py_files:
                    # If _v2 version files are not found, try to find old version
                    py_files = [f for f in os.listdir(path) if f.startswith(str(simulateidx)) and f.endswith('.py')]
            else:
                py_files = [f for f in os.listdir(path) if f.startswith(str(simulateidx)) and f.endswith('.py')]
            if not py_files:
                print(f"[INS_ERR] {interface_type} code not found: {path}")
                return
            exec_file = os.path.join(path, py_files[0])
        elif interface_type == 'dumb_interface':
            path = os.path.join(os.path.dirname(INSPECTION_DIR),
                                 'simulation', 'dumb_simulator', self.params['name'])
            api_name = self.params['api_name']
            #py_files = [f for f in os.listdir(path) if f.endswith('.py') and f.startswith(api_name)]
            #Use the new version of dumbsimulator code, filenames end with injected_.py
            py_files = [f for f in os.listdir(path) if f.endswith('injected.py') and f.startswith(api_name)]
            if not py_files:
                py_files = [f for f in os.listdir(path) if f.endswith('.py') and f.startswith(api_name)]
            if not py_files:
                print(f"[INS_ERR] {interface_type} code not found: {path}")
                return
            if len(py_files) > 1:
                if self.params.get('idx') is not None:
                    use_v2 = CONFIG.get('dumb_use_v2', True)
                    py_files = [f for f in py_files if str(self.params['idx']) in f ]
                    if use_v2:
                        py_files = [f for f in py_files if "dumbV2" in f]
                    else:
                        py_files = [f for f in py_files if "dumb" in f]
                else:
                    print(f"[INS_WARN] Found multiple {interface_type} code")
                    for i, f in enumerate(py_files):
                        print(f"    {i+1}: {f}")
                    choice = input(f"[INS_INFO] Select {interface_type} code to execute or number: ")
                    if choice.isdigit() and int(choice)-1 < len(py_files) and int(choice)-1 >= 0:
                        choice = int(choice) - 1
                        py_files = [py_files[int(choice)]]
                    elif choice in py_files:
                        py_files = [choice]
                    else:
                        print(f"[INS_ERR] Invalid selection: {choice}")
                        return
            exec_file = os.path.join(path, py_files[0])
        # Unified code execution part
        if exec_file:
            syntaxFail = False
            fail = False
            err_msg = ""
            try:
                with open(exec_file, 'r') as f:
                    code = f.read()
                    code_record_floder = os.path.join(RECORD_PATH, 'code_record', self.params['name'])
                    if not os.path.exists(code_record_floder):
                        os.makedirs(code_record_floder)
                    now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
                    code_record_path = os.path.join(code_record_floder, f"{self.params['name']}_{interface_type}_{now}.txt")
                    with open(code_record_path, 'w') as code_f:
                        code_f.write(code)
                    if CONFIG.get('exec_use_subprocess', False):
                        run_as_module(exec_file)
                    else:
                        exec(code, globals())
            except SyntaxError as e:
                print(f"[INS_ERR] {interface_type} code syntax error: {e}")
                syntaxFail = True
                fail = True
                err_msg = e
            except IndentationError as e:
                print(f"[INS_ERR] {interface_type} code syntax error: {e}")
                syntaxFail = True
                fail = True
                err_msg = e
            except Exception as e:
                print(f"[INS_ERR] {interface_type} code execution failed: {e}")
                fail = True
                err_msg = e
            # 建议生成部分
            gen_suggestion = CONFIG.get('auto_suggest', False)
            if not syntaxFail and gen_suggestion:
                from Inspection.generator.suggestion_generator import SuggestionGenerator
                suggestion_generator = SuggestionGenerator(self.params['name'])
                suggestion_generator.set_base_ai(self.BaseAI)
                if interface_type == 'test_interface':
                    # 需提示用户输入以指定引用的call和欲检查的接口名
                    call_id = self.__select_call_id()
                    api_name = self.__select_api_name()
                    suggestion_generator.generate_suggestions(api_name=api_name, idx=call_id , simulate_type='test')
                elif interface_type == 'simulate_interface':
                    # 需提示用户输入以指定欲检查的接口
                    api_name = self.__select_api_name()
                    suggestion_generator.generate_suggestions(api_name=api_name, idx=self.params['idx'], simulate_type='simulation')
                elif interface_type == 'dumb_interface':
                    # 需要让用户指定引用的call
                    call_id = self.__select_call_id()
                    suggestion_generator.generate_suggestions(api_name=self.params['api_name'], idx=call_id, simulate_type='dumb')
            self.__recover_sys_path()
            if fail:
                raise Exception(err_msg)

    def __select_call_id(self):
        from Inspection.utils.interface_reader import InterfaceInfoReader
        interface_info_reader = InterfaceInfoReader(self.params['name'])
        call_list = interface_info_reader.get_call_list()
        call_names = [call['Name'] for call in call_list]
        print(f"[INS_INFO] Available code list to inspect: ")
        for i, name in enumerate(call_names):
            print(f"    {i+1}: {name}")
        choice = input(f"[INS_INFO] Select code number to inspect: ")
        if choice.isdigit() and int(choice)-1 < len(call_names) and int(choice)-1 >= 0:
            choice = int(choice) - 1
        else:
            print(f"[INS_ERR] Invalid selection: {choice}")
            return
        return choice

    def __select_api_name(self):
        from Inspection.core.code_processor import extract_function_names_from_adapter
        adapter_data = get_class_code(CUSTOM_ADAPTER_PATH+self.params['name']+'.py', 'CustomAdapter')
        avaliable_fuc = extract_function_names_from_adapter(adapter_data)
        print(f"[INS_INFO] Available interface list:")
        for i, name in enumerate(avaliable_fuc):
            print(f"    {i+1}: {name}")
        choice = input(f"[INS_INFO] Select interface number to inspect: ")

        if choice.isdigit() and int(choice)-1 < len(avaliable_fuc) and int(choice)-1 >= 0:
            choice = int(choice) - 1
            choice = avaliable_fuc[choice]
        elif choice in avaliable_fuc:
            choice = choice
        else:
            print(f"[INS_ERR] Invalid selection: {choice}")
            return
        
        return choice

class Workflow:
    def __init__(self, simulate_pj_name: str, sys_additional_path: List, base_ai):
        self.BaseAI = base_ai
        self.now = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
        self.BaseAI.id = f"AIRecord_{simulate_pj_name}_{self.now}"
        self.simulate_pj_name = simulate_pj_name
        for p in sys_additional_path:
            if p not in sys.path:
                sys.path.append(p)
        self.steps : List[Step] = []
        self.current_step = 0

    def write_error_log(self, error_message):
        if CONFIG.get('evaluate_mode', False):
            err_path_base = os.path.join(EVALUATE_RQ1_PATH, 'error_logs')
        else:
            err_path_base = RECORD_PATH + "/error/"
        if not os.path.exists(err_path_base):
            os.makedirs(err_path_base)
        err_name = f"err_{self.simulate_pj_name}_{self.now}.txt"
        err_path = os.path.join(err_path_base, err_name)
        with open(err_path, 'a') as f:
            f.write(f"[INS_ERR] {error_message}\n")
        print(f"[INS_ERR] {error_message}")
        print(f"[INS_ERR] Error message has been logged to: {err_path}")
    
    def add_step(self, step_name , simulate_idx = None , api_name = None):
        step_params = {'name': self.simulate_pj_name}
        if step_name == 'simulation':
            step_params['idx'] = simulate_idx
        if step_name == 'generate_dumb_simulator':
            if api_name is None or simulate_idx is None:
                raise ValueError("generate_dumb_simulator step requires api_name and simulate_idx")
            step_params['idx'] = simulate_idx
            step_params['api_name'] = api_name
        if step_name == 'dumb':
            if api_name is None:
                raise ValueError("dumb step requires api_name")
            step_params['api_name'] = api_name
            if simulate_idx is not None:
                step_params['idx'] = simulate_idx
        if step_name == 'generate_simulation':
            if simulate_idx is None:
                print(f"[INS_WARN] generate_simulation step requires simulate_idx")
            step_params['idx'] = simulate_idx
        step = Step(step_name, **step_params)
        step.set_base_ai(self.BaseAI.copy())
        self.steps.append(step)


    def run(self):
        for step in self.steps:
            try:
                step.execute()
            except Exception as e:
                import traceback
                print(f"[INS_ERR] Step execution failed: {step.step_name}")
                traceback_info = traceback.format_exc()
                self.write_error_log(f"Step execution failed: {step.step_name}, error message: {e}\nStack trace: \n{traceback_info}")
            self.current_step += 1
    
    def run_step(self, step_name , simulate_idx = None , api_name = None):
        try:
            step = Step(step_name, name=self.simulate_pj_name, 
                        idx=simulate_idx if step_name == 'simulation' or step_name== 'generate_dumb_simulator' or step_name == 'generate_simulation' else None,
                        api_name=api_name if step_name == 'generate_dumb_simulator' or step_name == 'dumb' else None
                        )
            step.set_base_ai(self.BaseAI.copy())
            step.execute()
        except Exception as e:
            import traceback
            print(f"[INS_ERR] Step execution failed: {step_name}")
            traceback_info = traceback.format_exc()
            self.write_error_log(f"Step execution failed: {step_name}, error message: {e}\nStack trace: \n{traceback_info}")
    
    def clear_steps(self):
        self.steps.clear()
        self.current_step = 0



if __name__ == "__main__":
    pass