from Inspection.utils.path_manager import INTERFACE_INFO_PATH, INTERFACE_DOC_PATH ,CUSTOM_ADAPTER_PATH ,SIMULATION_PATH ,WORKFLOW_PATH ,INTERFACE_TXT_PATH
from Inspection.utils.config import CONFIG

from Inspection.core.code_processor import get_class_code
from Inspection.core.workflow import Workflow
from Inspection.utils.file_lister import FileLister
from Inspection.utils.workflow_compiler import WorkflowCompiler

from pathlib import Path
import importlib
import time

sys_path = CONFIG.get('path', [])

module = None
BaseAI = None


def modify_ai(use : bool):
    global BaseAI
    if use:
        module = importlib.import_module('Inspection.ai.base_ai')
        BaseAI = getattr(module, 'BaseAI')
    else:
        module = importlib.import_module('Inspection.ai.dead_base_ai')
        BaseAI = getattr(module, 'BaseAI')


def show_help():
    print("Help information:")
    print("    generate - Enter generation mode")
    print("    execute - Enter execution mode")
    print("    write - Write interface information")
    print("    read - Enter reading mode")
    print("    workflow - Enter workflow mode")
    print("    backup - Enter backup mode")
    print("    clean - Enter cleanup mode")
    print("    edit - Edit interface information file")
    print("    chat - Chat with AI")
    print("    modify_config - Modify configuration file")
    print("    quit - Exit program")
    print("    help - Show help information")
    print("    Shortcuts: g -> generate, e -> execute, w -> write, r -> read, wf -> workflow, b -> backup, cl -> clean, ed -> edit, ch -> chat, mc -> modify_config, q -> quit, h -> help")


def show_generate_help():
    print("Generation mode help information:")
    print("    doc - Generate interface documentation")
    print("    adapter - Generate adapters")
    # print("    test - Generate test execution code")
    print("    simulate - Generate static executable artifacts")
    print("    dumb - Generate non-intelligent module simulation functions")
    print("    quit - Exit generation mode")
    print("    help - Show help information")
    print("    Shortcuts: do -> doc, a -> adapter, s -> simulate, du -> dumb, q -> quit, h -> help")

def show_execute_help():
    print("Execution mode help information:")
    # print("    test - Execute test execution code")
    print("    simulate - Execute static artifacts")
    print("    dumb - Execute non-intelligent module simulation parameter injection scripts")
    print("    quit - Exit execution mode")
    print("    help - Show help information")
    print("    Shortcuts: s -> simulate, d -> dumb, q -> quit, h -> help")

def show_read_help():
    print("Reading mode help information:")
    print("    record - View execution records")
    print("    interface_info - View interface information")
    print("    interface_doc - View interface documentation")
    print("    quit - Exit reading mode")
    print("    help - Show help information")
    print("    Shortcuts: r -> record, ii -> interface_info, id -> interface_doc, q -> quit, h -> help")

def show_backup_help():
    print("Backup mode help information:")
    print("   interface_info - Backup interface information")
    print("   interface_doc - Backup interface documentation")
    print("   adapter - Backup adapters")
    print("   test - Backup test execution code")
    print("   simulate - Backup simulation execution code")
    print("   all - Backup all")
    print("   quit - Exit backup mode")
    print("   help - Show help information")
    print("   Shortcuts: ii -> interface_info, id -> interface_doc, a -> adapter, t -> test, s -> simulate, q -> quit, h -> help")

def show_clean_help():
    print("Cleanup mode help information:")
    print("   interface_info - Clean interface information")
    print("   interface_doc - Clean interface documentation")
    print("   adapter - Clean adapters")
    print("   test - Clean test execution code")
    print("   simulate - Clean simulation execution code")
    print("   single - Clean all related files for a single project")
    print("   dumb - Clean non-intelligent module simulation code")
    print("   quit - Exit cleanup mode")
    print("   help - Show help information")
    print("   Shortcuts: ii -> interface_info, id -> interface_doc, a -> adapter, t -> test, s -> simulate , sg -> single, q -> quit, h -> help")

def generate():
    modify_ai(True)
    print("\nEntering generation mode")
    show_generate_help()
    while True:
        command = input("[Generation Mode] Please enter command: ")
        if command == "help" or command == "h":
            show_generate_help()
        elif command == "doc" or command == "do":
            generate_doc()
        elif command == "adapter" or command == "a":
            generate_adapter()
        elif command == "test" or command == "t":
            generate_test()
        elif command == "simulate" or command == "s":
            generate_simulate()
        elif command == "dumb" or command == "du":
            generate_dumb()
        elif command == "quit" or command == "q":
            print("Exiting generation mode")
            break
        else:
            print(f"Unknown command: {command}, please enter 'help' for assistance.")
            continue
        print("[INS_INFO] Generation completed")

def generate_doc():
    file_lister = FileLister(INTERFACE_INFO_PATH, 'json')
    file_lister.print_file_list("Available interface information files:")
    if not file_lister.file_list:
        print(f"[INS_ERR] No interface information files exist, please generate interface information first")
        return
    selected_file_name = file_lister.choose_file("Please enter the interface information file number or filename to generate documentation for: ")
    # yolox
    if selected_file_name is None:
        return
    workflow = Workflow(selected_file_name, sys_path , BaseAI())
    workflow.run_step('generate_doc')

def generate_adapter():
    filelister= FileLister(INTERFACE_DOC_PATH, 'md')
    filelister.print_file_list("Available interface documentation:")
    if not filelister.file_list:
        print(f"[INS_ERR] No interface documentation exists, please generate interface documentation first")
        return
    selected_file_name = filelister.choose_file("Please enter the interface documentation number or filename to generate adapter for: ")
    if selected_file_name is None:
        return
    workflow = Workflow(selected_file_name, sys_path , BaseAI())
    workflow.run_step('generate_adapter')

def generate_test():
    md_file_lister = FileLister(INTERFACE_DOC_PATH, 'md')
    py_file_lister = FileLister(CUSTOM_ADAPTER_PATH, 'py')
    if md_file_lister.file_list == []:
        print(f"[INS_ERR] No interface documentation exists, please generate interface documentation first")
        return
    if py_file_lister.file_list == []:
        print(f"[INS_ERR] No adapters exist, please generate adapters first")
        return
    common_files = set(md_file_lister.file_list) & set(py_file_lister.file_list)
    filelister = FileLister()
    filelister.file_list = list(common_files)
    filelister.print_file_list("Available project list")
    selected_file_name = filelister.choose_file("Please enter the number or filename to generate test execution code for: ")
    if selected_file_name is None:
        return
    workflow = Workflow(selected_file_name, sys_path , BaseAI())
    workflow.run_step('generate_test')


def generate_simulate():
    md_file_lister = FileLister(INTERFACE_DOC_PATH, 'md')
    if md_file_lister.file_list == []:
        print(f"[INS_ERR] No interface documentation exists, please generate interface documentation first")
        return
    py_file_lister = FileLister(CUSTOM_ADAPTER_PATH, 'py')
    if py_file_lister.file_list == []:
        print(f"[INS_ERR] No adapters exist, please generate adapters first")
        return
    json_file_lister = FileLister(INTERFACE_INFO_PATH, 'json')
    if json_file_lister.file_list == []:
        print(f"[INS_ERR] No interface information exists, please generate interface information first")
        return
    common_files = set(md_file_lister.file_list) & set(py_file_lister.file_list) & set(json_file_lister.file_list)
    filelister = FileLister()
    filelister.file_list = list(common_files)
    filelister.print_file_list("Available project list")
    selected_file_name = filelister.choose_file("Please enter the project number or filename to generate static executable artifacts for: ")
    if selected_file_name is None:
        return
    pj_name = selected_file_name
    from Inspection.utils.interface_reader import InterfaceInfoReader
    interface_info_reader = InterfaceInfoReader(pj_name)
    call_list = interface_info_reader.get_call_list()
    name_list = [data['Name'] for data in call_list]
    filelister = FileLister()
    filelister.file_list = name_list
    filelister.print_file_list("Available call data:")
    choice = filelister.choose_file("Please enter the call data number or filename to be rebuilt: ")
    if choice is None:
        return
    selected_id = -1
    for i, data in enumerate(name_list):
        if data == choice:
            selected_id = i
            break
    if selected_id == -1:
        print(f"[INS_ERR] Invalid selection: {choice}")
        return
    workflow = Workflow(pj_name, sys_path , BaseAI())
    workflow.run_step('generate_simulation' , simulate_idx=selected_id)

def generate_dumb():
    md_file_lister = FileLister(INTERFACE_DOC_PATH, 'md')
    if md_file_lister.file_list == []:
        print(f"[INS_ERR] No interface documentation exists, please generate interface documentation first")
        return
    py_file_lister = FileLister(CUSTOM_ADAPTER_PATH, 'py')
    if py_file_lister.file_list == []:
        print(f"[INS_ERR] No adapters exist, please generate adapters first")
        return
    json_file_lister = FileLister(INTERFACE_INFO_PATH, 'json')
    if json_file_lister.file_list == []:
        print(f"[INS_ERR] No interface information exists, please generate interface information first")
        return
    common_files = set(md_file_lister.file_list) & set(py_file_lister.file_list) & set(json_file_lister.file_list)
    filelister = FileLister()
    filelister.file_list = list(common_files)
    filelister.print_file_list("Available project list")
    selected_file_name = filelister.choose_file("Please select project number or filename: ")
    if selected_file_name is None:
        return
    from Inspection.core.code_processor import extract_function_names_from_adapter
    adapter_data = get_class_code(CUSTOM_ADAPTER_PATH+selected_file_name+'.py', 'CustomAdapter')
    avaliable_fuc = extract_function_names_from_adapter(adapter_data)
    filelister = FileLister()
    filelister.file_list = avaliable_fuc
    filelister.print_file_list("Available smart module interfaces for simulating input parameters:")
    selected_fuc_name = filelister.choose_file("Please select interface number or filename: ")
    if selected_fuc_name is None:
        return
    from Inspection.utils.interface_reader import InterfaceInfoReader
    interface_info_reader = InterfaceInfoReader(selected_file_name)
    calldata = interface_info_reader.get_call_list()
    call_name_list = [call['Name'] for call in calldata]
    filelister.file_list = call_name_list
    filelister.print_file_list("Available call data:")
    choice = filelister.choose_file("Please enter the call data number or filename to be simulated for non-smart modules: ")
    selected_id = -1
    for i, data in enumerate(calldata):
        if data['Name'] == choice:
            selected_id = i
            break
    if selected_id == -1:
        print(f"[INS_ERR] Invalid selection: {choice}")
        return

    workflow = Workflow(selected_file_name, sys_path , BaseAI())
    workflow.run_step('generate_dumb_simulator', api_name=selected_fuc_name, simulate_idx=selected_id)
        


def execute():
    # Execute mode
    auto_suggest = CONFIG.get('auto_suggest', False)
    if not auto_suggest:
        modify_ai(False)
    else:
        modify_ai(True)
    print("\nEntering execute mode")
    show_execute_help()
    while True:
        command = input("[Execute Mode] Please enter command: ")
        if command == "help" or command == "h":
            show_execute_help()
        elif command == "quit" or command == "q":
            print("Exiting execute mode")
            break
        elif command == "test" or command == "t":
            execute_test()
        elif command == "simulate" or command == "s":
            execute_simulate()
        elif command == "dumb" or command == "d":
            execute_dumb()
        else:
            print(f"Unknown command: {command}, please type help for assistance.")
            continue
        print("[INS_INFO] Execution completed")

def execute_test():
    filelister = FileLister(SIMULATION_PATH + "/test_interface", 'py')
    filelister.print_file_list("Available test execution code:")
    if not filelister.file_list:
        print(f"[INS_ERR] No test execution code exists, please generate test execution code first")
        return
    selected_file_name = filelister.choose_file("Please enter the test execution code number or filename to execute: ")
    if selected_file_name is None:
        return
    workflow = Workflow(selected_file_name, sys_path , BaseAI())
    try:
        workflow.run_step('test')
    except:
        print(f"[INS_ERR] Execution failed, please manually check if adapter code and test execution code are correct")
        print(f"[INS_INFO] Adapter location: {CUSTOM_ADAPTER_PATH}{selected_file_name}.py")
        print(f"[INS_INFO] Test execution code location: {SIMULATION_PATH}/test_interface/{selected_file_name}.py")


def execute_simulate():
    filelister = FileLister(SIMULATION_PATH + "/simulate_interface", 'dir' , not_include = "backup_")
    filelister.print_file_list("Available projects:")
    if not filelister.file_list:
        print(f"[INS_ERR] No available projects exist, please generate static executable artifacts first")
        return
    selected_dir_name = filelister.choose_file("Please enter the project number or folder name to execute: ")
    if selected_dir_name is None:
        return
    selected_dir = SIMULATION_PATH + "/simulate_interface/" + selected_dir_name
    # Enter the folder and find all .py files
    pyfilelister = FileLister(selected_dir, 'py')
    # For each filename in filelister.file_list, remove _v2 suffix if it exists
    templist = []
    for file in pyfilelister.file_list:
        if file.endswith("_v2"):
            templist.append(file[:-3])
        else:
            templist.append(file)
    # Remove duplicate filenames
    templist = list(set(templist))
    pyfilelister.file_list = templist
    pyfilelister.print_file_list("Available static executable artifacts:")
    if not pyfilelister.file_list:
        print(f"[INS_ERR] No static executable artifacts exist, please generate static executable artifacts first")
        return
    selected_file_name = pyfilelister.choose_file("Please enter the static artifact number or filename to execute: ")
    if selected_file_name is None:
        return
    simulate_idx = selected_file_name.split("_")[0]
    workflow = Workflow(selected_dir_name, sys_path , BaseAI())
    try:
        workflow.run_step('simulation', simulate_idx=simulate_idx)
    except Exception as e:
        print(e)
        print(f"[INS_ERR] Execution failed, please manually check if adapter code and static artifact code are correct")
        print(f"[INS_INFO] Adapter location: {CUSTOM_ADAPTER_PATH}{selected_file_name}.py")
        print(f"[INS_INFO] Simulation execution code location: {SIMULATION_PATH}simulate_interface/{selected_dir_name}/{selected_file_name}.py")

def execute_dumb():
    filelister = FileLister(SIMULATION_PATH + "/dumb_simulator", 'dir' , not_include = "backup_")
    filelister.print_file_list("Available non-intelligent module simulation projects:")
    if not filelister.file_list:
        print(f"[INS_ERR] No non-intelligent module simulation projects exist, please generate non-intelligent module simulation projects first")
        return
    selected_dir_name = filelister.choose_file("Please enter the non-intelligent module simulation project number or folder name to execute: ")
    if selected_dir_name is None:
        return
    selected_dir = SIMULATION_PATH + "/dumb_simulator/" + selected_dir_name
    pyfiles = list(Path(selected_dir).glob("*.py"))
    #获取每个文件去除_call(id)_dumb.py后缀的字符串
    call_names = [file.stem.split("_call")[0] for file in pyfiles]
    # 去重
    call_names = list(set(call_names))
    # 选择文件
    filelister = FileLister()
    filelister.file_list = call_names
    filelister.print_file_list("Available non-intelligent module simulation interfaces:")
    choice = filelister.choose_file("Please enter the non-intelligent module simulation interface number or filename to execute: ")
    call_name = choice
    workflow = Workflow(selected_dir_name, sys_path , BaseAI())
    try:
        workflow.run_step('dumb', api_name=call_name)
    except Exception as e:
        print(e)
        print(f"[INS_ERR] Execution failed, please manually check if adapter code and simulation code are correct")
        print(f"[INS_INFO] Adapter location: {CUSTOM_ADAPTER_PATH}{selected_dir_name}.py")
        print(f"[INS_INFO] Non-intelligent module simulation code location: {SIMULATION_PATH}/dumb_simulator/{selected_dir_name}")

def workflow():
    # 工作流模式
    modify_ai(True)
    print("\nEntering workflow mode")
    filelister = FileLister(WORKFLOW_PATH, 'txt')
    filelister.print_file_list("Available workflow files:")
    if not filelister.file_list:
        print(f"[INS_ERR] No workflow files exist, please generate workflow files first")
        return
    selected_file_name = filelister.choose_file("Please enter the workflow file number or filename to execute: ")
    if selected_file_name is None:
        return
    workflowCompiler = WorkflowCompiler(WORKFLOW_PATH + selected_file_name + '.txt', BaseAI())
    workflow = workflowCompiler.compile()
    if workflow is None:
        print(f"[INS_ERR] Workflow compilation failed, please check workflow file")
        return
    workflow.run()
    print(f"[INS_INFO] Workflow execution completed")
    return



def read():
    print("\nEntering read mode")
    # 读取模式
    show_read_help()
    while True:
        command = input("[Read Mode] Please enter command: ")
        if command == "help" or command == "h":
            show_read_help()
        elif command == "quit" or command == "q":
            print("Exiting read mode")
            break
        elif command == "record" or command == "r":
            read_record()
        elif command == "interface_info" or command == "ii":
            read_interface_info()
        elif command == "interface_doc" or command == "id":
            read_interface_doc()
        else:
            print(f"Unknown command: {command}, please type help for assistance.")

def read_record():
    from Inspection.utils.result_reader import main as read_record_main
    read_record_main()

def read_interface_info():
    from Inspection.utils.interface_reader import InterfaceInfoReader
    # 找到接口信息文件
    interface_info_path = Path(INTERFACE_INFO_PATH)
    interface_info_files = list(interface_info_path.glob("*.json"))
    # 打印文件列表
    print("[INS_INFO] Available interface information files:")
    for i, file in enumerate(interface_info_files):
        print(f"{i + 1}. {file.stem}")
    # 选择文件
    choice = input("Please enter the interface information file number or filename to view: ")
    if choice.isdigit():
        choice = int(choice) - 1
        if 0 <= choice < len(interface_info_files):
            selected_file = interface_info_files[choice]
        else:
            print(f"[INS_ERR] Invalid selection: {choice + 1}")
            return
    else:
        if choice.endswith(".json"):
            selected_file = interface_info_path / choice
        else:
            selected_file = interface_info_path / (choice + ".json")
        if not selected_file.exists():
            print(f"[INS_ERR] File does not exist: {selected_file}")
            return
    filename = selected_file.stem
    # 读取文件
    info_reader = InterfaceInfoReader(filename)
    info_reader.print_info()

def read_interface_doc():
    from Inspection.utils.interface_reader import InterfaceDocReader
    # 找到接口文档
    interface_doc_path = Path(INTERFACE_DOC_PATH)
    interface_doc_files = list(interface_doc_path.glob("*.md"))
    # 打印文件列表
    print("[INS_INFO] Available interface documentation:")
    for i, file in enumerate(interface_doc_files):
        print(f"{i + 1}. {file.stem}")
    # 选择文件
    choice = input("Please enter the interface documentation number or filename to view: ")
    if choice.isdigit():
        choice = int(choice) - 1
        if 0 <= choice < len(interface_doc_files):
            selected_file = interface_doc_files[choice]
        else:
            print(f"[INS_ERR] Invalid selection: {choice + 1}")
            return
    else:
        if choice.endswith(".md"):
            selected_file = interface_doc_path / choice
        else:
            selected_file = interface_doc_path / (choice + ".md")
        if not selected_file.exists():
            print(f"[INS_ERR] File does not exist: {selected_file}")
            return
    filename = selected_file.stem
    # 读取文件
    doc_reader = InterfaceDocReader(filename)
    print(doc_reader.get_doc())

def backup():
    print("\nEntering backup mode")
    show_backup_help()
    from Inspection.utils.backup import backup_all , backup_Custom_adapter , backup_Interface_doc
    from Inspection.utils.backup import backup_Interface_info , backup_simulation_code , backup_test_code
    while True:
        command = input("[Backup Mode] Please enter command: ")
        if command == "help" or command == "h":
            show_backup_help()
        elif command == "quit" or command == "q":
            print("Exiting backup mode")
            break
        elif command == "interface_info" or command == "ii":
            backup_Interface_info()
        elif command == "interface_doc" or command == "id":
            backup_Interface_doc()
        elif command == "adapter" or command == "a":
            backup_Custom_adapter()
        elif command == "test" or command == "t":
            backup_test_code()
        elif command == "simulate" or command == "s":
            backup_simulation_code()
        elif command == "all":
            backup_all()
        else:
            print(f"Unknown command: {command}, please type help for assistance.")

def clean():
    print("\nEntering clean mode (please use with caution)")
    show_clean_help()
    from Inspection.utils.clean import clean_Interface_info , clean_Interface_doc
    from Inspection.utils.clean import clean_Custom_adapter , clean_test_code , clean_simulation_code
    while True:
        command = input("[Clean Mode] Please enter command: ")
        if command == "help" or command == "h":
            show_clean_help()
        elif command == "quit" or command == "q":
            print("Exiting clean mode")
            break
        elif command == "interface_info" or command == "ii":
            clean_Interface_info()
        elif command == "interface_doc" or command == "id":
            clean_Interface_doc()
        elif command == "adapter" or command == "a":
            clean_Custom_adapter()
        elif command == "test" or command == "t":
            clean_test_code()
        elif command == "simulate" or command == "s":
            clean_simulation_code()
        elif command == "single" or command == "sg":
            filelister = FileLister(INTERFACE_INFO_PATH, 'json')
            filelister.print_file_list("Available project files:")
            if not filelister.file_list:
                print(f"[INS_ERR] No project files exist, please generate interface information first")
                return
            selected_file_name = filelister.choose_file("Please enter the project file number or filename to clean: ")
            if selected_file_name is None:
                return
            from Inspection.utils.clean import clean_single_project
            clean_single_project(selected_file_name)
        # 过于危险的操作，暂时不提供
        # elif command == "all":
        #     clean_Interface_info()
        #     clean_Interface_doc()
        #     clean_Custom_adapter()
        #     clean_test_code()
        #     clean_simulation_code()
        else:
            print(f"Unknown command: {command}, please type help for assistance.")

def write():
    print("Write mode, can format and write interface information from InterfaceTXT directory to interface information files")
    from Inspection.utils.interface_writer import InterfaceWriter
    name = input("[Write Mode] Please name the interface information file (no extension needed): ")
    interfacewriter = InterfaceWriter(name)
    if(interfacewriter.check_info_exist()):
        print(f"[INS_WARN] Interface information file already exists, add interface information or overwrite interface information?")
        cover = input("[Write Mode] (Add y/Overwrite n): ")
        if cover == 'y':
            interfacewriter.cover = False
            print("[INS_INFO] Please select append mode:\n1. Add only call information\n2. Add only implementation information\n3. Add all")
            choice = input("[Write Mode] Please enter number to select: ")
            if choice == '1':
                interfacewriter.write(calls=True, ipls=False)
                print(f"[INS_INFO] Added call information to {name}.json")
            elif choice == '2':
                interfacewriter.write(calls=False, ipls=True)
                print(f"[INS_INFO] Added implementation information to {name}.json")
            elif choice == '3':
                interfacewriter.write(calls=True, ipls=True)
                print(f"[INS_INFO] Added information to {name}.json")
            else:
                print(f"[INS_ERR] Input error, exiting")
                return
        elif cover == 'n':
            interfacewriter.cover = True
            interfacewriter.write()
            print(f"[INS_INFO] Interface information file has been overwritten to {name}.json")
        else:
            print(f"[INS_ERR] Input error, exiting")
            return
    else:
        interfacewriter.cover = True
        interfacewriter.write()
        print(f"[INS_INFO] Interface information file has been successfully written to {name}.json")


def edit():
    print("Edit mode, can edit interface information files")
    from Inspection.utils.interface_writer import InterfaceRewriter,InterfaceWriter
    from Inspection.utils.file_lister import FileLister
    filelister = FileLister(INTERFACE_INFO_PATH, 'json')
    filelister.print_file_list("Available interface information files:")
    if not filelister.file_list:
        print(f"[INS_ERR] No interface information files exist, please generate interface information first")
        return
    selected_file_name = filelister.choose_file("Please enter the interface information file number or filename to edit: ")
    if selected_file_name is None:
        return
    interface_rewriter = InterfaceRewriter(selected_file_name)
    interface_rewriter.rewrite()
    print(f"[INS_INFO] Interface information file has been successfully rewritten to  {INTERFACE_TXT_PATH}  directory")
    print(f"[INS_INFO] Operation completed, please enter 'ok' to write back, or enter 'quit' to abandon")
    while True:
        command = input("[Edit Mode] Please enter command: ")
        if command == "ok":
            print("[INS_INFO] Writing back interface information file")
            interface_writer = InterfaceWriter(selected_file_name, cover=True)
            interface_writer.write()
            print(f"[INS_INFO] Interface information file has been successfully written back to  {INTERFACE_INFO_PATH} 下的 {selected_file_name}.json")
            break
        elif command == "quit":
            print("[INS_INFO] 放弃Writing back interface information file，操作已取消")
            return
        else:
            print(f"[INS_ERR] Invalid command: {command}, please enter 'ok' or 'quit'")

def modify_config():
    global CONFIG
    print("Current project configuration:")
    for key, value in CONFIG.items():
        if key == '/*comment*/':
            continue
        print(f"    {key}: {value}")
    print("Please enter the configuration item to modify (or enter 'quit' to exit):")
    while True:
        command = input("[Modify Config] Please enter command: ")
        if command == 'quit' or command == 'q':
            print("Exiting configuration modification")
            break
        elif command in CONFIG:
            new_value = input(f"Please enter new value (current value is {CONFIG[command]}): ")
            if new_value:
                if new_value.lower() == 'true':
                    new_value = True
                elif new_value.lower() == 'false':
                    new_value = False
                elif new_value.isdigit():
                    new_value = int(new_value)
                elif new_value.replace('.','',1).isdigit() and new_value.count('.') < 2:
                    new_value = float(new_value)
                CONFIG[command] = new_value
                print(f"[INS_INFO] Changed {command} to {new_value}")
            else:
                print(f"[INS_WARN] Did not modify {command}'s value")
        else:
            print(f"[INS_ERR] Invalid configuration item: {command}, please re-enter or type 'exit' to exit")


def chat_with_ai():
    modify_ai(True)
    print("Entering AI chat mode")
    print("Please enter your question, type 'quit' to exit chat mode")
    ai = BaseAI(id = 'Chat_log_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    while True:
        question = input("[User] : ")
        if question.lower() == 'quit':
            print("Exiting AI chat mode")
            break
        elif question.strip() == '':
            continue
        else:
            # 调用AI进行回答
            ai_response = ai.generate_text(question)
            print(f"[AI] : {ai_response}\n")


def cli():
    modify_ai(False)
    print("Welcome to Inspection CLI tool!")
    if CONFIG.get('evaluate_mode', False):
        print('Currently in evaluation mode')
    show_help()
    while True:
        command = input("[Inspection CLI] Please enter command: ")
        if command == "help" or command == "h":
            show_help()
        elif command == "quit" or command == "q":
            print("Thank you for using Inspection CLI tool!")
            break
        elif command == "generate" or command == "g":
            generate()
        elif command == "execute" or command == "e":
            execute()
        elif command == "read" or command == "r":
            read()
        elif command == "backup" or command == "b":
            backup()
        elif command == "clean" or command == "cl":
            clean()
        elif command == "write" or command == "w":
            write()
        elif command == "workflow" or command == "wf":
            workflow()
        elif command == "edit" or command == "ed":
            edit()
        elif command == "modify_config" or command == "mc":
            modify_config()
        elif command == "chat" or command == "ch":
            chat_with_ai()
        else:
            print(f"Unknown command: {command}, please type help for assistance.")
