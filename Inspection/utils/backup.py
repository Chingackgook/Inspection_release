from Inspection import INTERFACE_DOC_PATH , INTERFACE_INFO_PATH
from Inspection import CUSTOM_ADAPTER_PATH
from Inspection import SIMULATION_PATH

import os
import shutil
from datetime import datetime
from pathlib import Path

def pack_files(source_dir , file_extension):
    # 获取当前时间戳（格式：YYYYMMDD_HHMMSS）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = "backup_" + timestamp
    
    # 创建目标文件夹路径（在源目录下创建）
    target_dir = Path(source_dir) / name
    target_dir.mkdir(exist_ok=True)  # 自动创建目录
    
    # 遍历源目录中的所有文件
    for file_path in Path(source_dir).glob(f"*{file_extension}"):
        try:
            # 构建目标文件路径
            relative_path = file_path.relative_to(source_dir)
            target_path = target_dir / relative_path
            
            # 创建目标子目录（保持原目录结构）
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件（保留元数据）
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")

def backup_Interface_info():
    print("[INS_INFO] Backing up interface information")
    pack_files(INTERFACE_INFO_PATH, '.json')

def backup_Interface_doc():
    print("[INS_INFO] Backing up interface documentation")
    pack_files(INTERFACE_DOC_PATH, '.md')

def backup_Custom_adapter():
    print("[INS_INFO] Backing up custom adapters")
    pack_files(CUSTOM_ADAPTER_PATH, '.py')

def backup_test_code():
    print("[INS_INFO] Backing up test execution code")
    pack_files(SIMULATION_PATH+'test_interface', '.py')
    pack_files(SIMULATION_PATH+'test_interface', '.md')

def backup_simulation_code():
    print("[INS_INFO] Backing up simulation execution code")
    source_dir = Path(SIMULATION_PATH + 'simulate_interface')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = source_dir / f'backup_{timestamp}'
    backup_dir.mkdir(exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.startswith('backup_')]  # 跳过以 backup_ 开头的目录
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(source_dir)
            target_path = backup_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path}")

def backup_dumb_code():
    print("[INS_INFO] Backing up simulation execution code")
    source_dir = Path(SIMULATION_PATH + 'dumb_simulator')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = source_dir / f'backup_{timestamp}'
    backup_dir.mkdir(exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [d for d in dirs if not d.startswith('backup_')]  # 跳过以 backup_ 开头的目录
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(source_dir)
            target_path = backup_dir / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_path)
            print(f"Copied: {file_path}")

def backup_all():
    backup_Interface_info()
    backup_Interface_doc()
    backup_Custom_adapter()
    backup_test_code()
    backup_simulation_code()
    backup_dumb_code()
    print("[INS_INFO] Backup completed")

if __name__ == "__main__":
    backup_Interface_info()
    backup_Interface_doc()
    backup_Custom_adapter()
    backup_test_code()
    backup_simulation_code()
    backup_dumb_code()
    print("[INS_INFO] Backup completed")

