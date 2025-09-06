from pathlib import Path
from Inspection import INTERFACE_DOC_PATH , INTERFACE_INFO_PATH
from Inspection import CUSTOM_ADAPTER_PATH
from Inspection import SIMULATION_PATH
import os

import Inspection.utils.backup as Backup



all_insure = False


def insure():
    global all_insure
    if all_insure:
        return True
    choice = input("Please confirm again whether to execute this command(Y/YALL/N): ").strip().upper()
    if choice == 'Y':
        print("Executing clean command...")
        return True
    elif choice == 'YALL':
        print("Executing clean command...")
        all_insure = True
        return True
    else:
        print("Clean command cancelled.")
        return False

def delete_files(source_dir, file_extension):
    """
    删除指定目录下的所有指定扩展名的文件
    :param source_dir: 源目录路径
    :param file_extension: 文件扩展名（例如：'.py'）
    """
    # 遍历源目录中的所有文件
    for file_path in Path(source_dir).glob(f"*{file_extension}"):
        try:
            # 删除文件
            file_path.unlink()
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")

def clean_Interface_info():
    if not insure():
        return
    Backup.backup_Interface_info()
    print("[INS_INFO] Cleaning interface information")
    delete_files(INTERFACE_INFO_PATH, '.json')

def clean_Interface_doc():
    if not insure():
        return
    Backup.backup_Interface_doc()
    print("[INS_INFO] Cleaning interface documentation")
    delete_files(INTERFACE_DOC_PATH, '.md')

def clean_Custom_adapter():
    if not insure():
        return
    Backup.backup_Custom_adapter()
    print("[INS_INFO] Cleaning custom adapters")
    delete_files(CUSTOM_ADAPTER_PATH, '.py')

def clean_test_code():
    if not insure():
        return
    Backup.backup_test_code()
    print("[INS_INFO] Cleaning test execution code")
    delete_files(SIMULATION_PATH+'test_interface', '.py')
    delete_files(SIMULATION_PATH+'test_interface', '.md')

def clean_simulation_code():
    if not insure():
        return
    Backup.backup_simulation_code()
    print("[INS_INFO] Cleaning simulation execution code")
    for root, dirs, files in os.walk(SIMULATION_PATH + 'simulate_interface'):
        dirs[:] = [d for d in dirs if not d.startswith('backup_')]  # 跳过以 backup_ 开头的目录
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {str(e)}")

    # 递归删除空目录（排除 'backup_' 开头的目录）
    for root, dirs, files in os.walk(SIMULATION_PATH + 'simulate_interface', topdown=False):
        for d in dirs:
            if d.startswith('backup_'):
                continue
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted empty directory: {dir_path}")
                except Exception as e:
                    print(f"Failed to process {dir_path}: {str(e)}")

def clean_dumb_code():
    if not insure():
        return
    Backup.backup_dumb_code()
    for root, dirs, files in os.walk(SIMULATION_PATH + 'dumb_simulator'):
        dirs[:] = [d for d in dirs if not d.startswith('backup_')]  # 跳过以 backup_ 开头的目录
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {str(e)}")

    # 递归删除空目录（排除 'backup_' 开头的目录）
    for root, dirs, files in os.walk(SIMULATION_PATH + 'dumb_simulator', topdown=False):
        for d in dirs:
            if d.startswith('backup_'):
                continue
            dir_path = os.path.join(root, d)
            if not os.listdir(dir_path):
                try:
                    os.rmdir(dir_path)
                    print(f"Deleted empty directory: {dir_path}")
                except Exception as e:
                    print(f"Failed to process {dir_path}: {str(e)}")


def clean_single_project(project_name):
    if not insure():
        return
    going_delete_doc_path = INTERFACE_DOC_PATH + project_name + '.md'
    going_delete_info_path = INTERFACE_INFO_PATH + project_name + '.json'
    going_delete_adapter_path = CUSTOM_ADAPTER_PATH + project_name + '.py'
    going_delete_simulate_dir = SIMULATION_PATH + 'simulate_interface/' + project_name
    going_delete_dumb_dir = SIMULATION_PATH + 'dumb_simulator/' + project_name
    if os.path.exists(going_delete_doc_path):
        os.remove(going_delete_doc_path)
        print(f"Deleted interface documentation: {going_delete_doc_path}")
    if os.path.exists(going_delete_info_path):
        os.remove(going_delete_info_path)
        print(f"Deleted interface information: {going_delete_info_path}")
    if os.path.exists(going_delete_adapter_path):
        os.remove(going_delete_adapter_path)
        print(f"Deleted custom adapter: {going_delete_adapter_path}")
    if os.path.exists(going_delete_simulate_dir):
        for root, dirs, files in os.walk(going_delete_simulate_dir, topdown=False):
            for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted simulation execution code: {file_path}")
                    except Exception as e:
                        print(f"Failed to process {file_path}: {str(e)}")
            for d in dirs:
                dir_path = os.path.join(root, d)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        print(f"Deleted empty directory: {dir_path}")
                    except Exception as e:
                        print(f"Failed to process {dir_path}: {str(e)}")
        os.rmdir(going_delete_simulate_dir)
    if os.path.exists(going_delete_dumb_dir):
        for root, dirs, files in os.walk(going_delete_dumb_dir, topdown=False):
            for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted non-intelligent module simulation code: {file_path}")
                    except Exception as e:
                        print(f"Failed to process {file_path}: {str(e)}")
            for d in dirs:
                dir_path = os.path.join(root, d)
                if not os.listdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                        print(f"Deleted empty directory: {dir_path}")
                    except Exception as e:
                        print(f"Failed to process {dir_path}: {str(e)}")
        os.rmdir(going_delete_dumb_dir)


def clean_all():
    clean_Interface_info()
    clean_Interface_doc()
    clean_Custom_adapter()
    clean_test_code()
    clean_simulation_code()
    clean_dumb_code()
    print("[INS_INFO] Cleanup completed")

if __name__ == "__main__":
    # clean_Interface_info()
    # clean_Interface_doc()
    # clean_Custom_adapter()
    # clean_test_code()
    clean_simulation_code()
    print("[INS_INFO] Cleanup completed")