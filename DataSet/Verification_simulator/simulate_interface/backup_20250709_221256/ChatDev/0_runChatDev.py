import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.ChatDev import ENV_DIR
from Inspection.adapters.custom_adapters.ChatDev import *

exe = Executor('ChatDev', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
import argparse
import logging
import os
import sys
from camel.typing import ModelType
from chatdev.chat_chain import ChatChain
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message import FunctionCall

# 模拟用户输入的参数
args = {
    'config': "Default",
    'org': "DefaultOrganization",
    'task': "Develop a basic Gomoku game.",
    'name': "Gomoku",
    'model': "GPT_4O_MINI",
    'path': ""
}

root = os.path.dirname(__file__)
sys.path.append(root)

try:
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message import FunctionCall

    openai_new_api = True  # new openai api version
except ImportError:
    openai_new_api = False  # old openai api version
    print(
        "Warning: Your OpenAI version is outdated. \n "
        "Please update as specified in requirement.txt. \n "
        "The old API interface is deprecated and will no longer be supported.")

def get_config(company):
    config_dir = os.path.join(ENV_DIR, "CompanyConfig", company)
    default_config_dir = os.path.join(ENV_DIR, "CompanyConfig", "Default")

    config_files = [
        "ChatChainConfig.json",
        "PhaseConfig.json",
        "RoleConfig.json"
    ]

    config_paths = []

    for config_file in config_files:
        company_config_path = os.path.join(config_dir, config_file)
        default_config_path = os.path.join(default_config_dir, config_file)

        if os.path.exists(company_config_path):
            config_paths.append(company_config_path)
        else:
            config_paths.append(default_config_path)

    return tuple(config_paths)

# Start ChatDev

# ----------------------------------------
#          Init ChatChain
# ----------------------------------------
config_path, config_phase_path, config_role_path = get_config(args['config'])
args2type = {
    'GPT_3_5_TURBO': ModelType.GPT_3_5_TURBO,
    'GPT_4': ModelType.GPT_4,
    'GPT_4_TURBO': ModelType.GPT_4_TURBO,
    'GPT_4O': ModelType.GPT_4O,
    'GPT_4O_MINI': ModelType.GPT_4O_MINI,
}

if openai_new_api:
    args2type['GPT_4O_MINI'] = ModelType.GPT_4O_MINI

# 调用 exe.create_interface_objects 进行初始化
exe.create_interface_objects(
    config_path=config_path,
    config_phase_path=config_phase_path,
    config_role_path=config_role_path,
    task_prompt=args['task'],
    project_name=args['name'],
    org_name=args['org'],
    model_type=args2type[args['model']],
    code_path=os.path.join(FILE_RECORD_PATH, args['path'])  # 使用 FILE_RECORD_PATH
)

# ----------------------------------------
#          Init Log
# ----------------------------------------
logging.basicConfig(filename=exe.adapter.chat_chain.log_filepath, level=logging.INFO,
                    format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%Y-%d-%m %H:%M:%S', encoding="utf-8")

# ----------------------------------------
#          Pre Processing
# ----------------------------------------
exe.run("pre_processing")

# ----------------------------------------
#          Personnel Recruitment
# ----------------------------------------
exe.run("make_recruitment")

# ----------------------------------------
#          Chat Chain
# ----------------------------------------
exe.run("execute_chain")

# ----------------------------------------
#          Post Processing
# ----------------------------------------
exe.run("post_processing")
