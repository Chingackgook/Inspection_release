
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.gpt_engineer import ENV_DIR
from Inspection.adapters.custom_adapters.gpt_engineer import *
exe = Executor('gpt_engineer','simulation')
FILE_RECORD_PATH = exe.now_record_path

# 假设 exe 是一个已经定义好的对象
# 请根据实际情况初始化 exe
exe.create_interface_objects(class_name='AI', model_name='gpt-4-turbo', temperature=0.1)  # 初始化模型，确保class_name为'AI'

# 基础参数输入
preprompts = {
    "roadmap": "这是一个示例 roadmap",
    "generate": "根据给定内容生成代码，格式为: FILE_FORMAT",
    "improve": "根据已有代码进行改进，格式为: FILE_FORMAT",
    "file_format": "Python",
    "file_format_diff": "Python差异",
    "philosophy": "保持简洁和高效",
    "entrypoint": "生成入口代码的提示"
}
prompt = Prompt(entrypoint_prompt="生成入口代码的提示内容",text="这是一个示例文本")
files_dict = FilesDict({"example.py": "print('Hello, World!')"})

# 开始会话
messages = exe.run("AI_start", 
                   system=setup_sys_prompt(preprompts), 
                   user=prompt.to_langchain_content(), 
                   step_name=curr_fn())

chat = messages[-1].content.strip()
files_dict = chat_to_files_dict(chat)

# 继续会话
updated_messages = exe.run("AI_next", 
                            messages=messages, 
                            prompt="这是一个新的提示",
                            step_name=curr_fn())

# 使用推理
response = exe.run("AI_backoff_inference", messages=messages)

# 序列化消息
json_string = exe.run("AI_serialize_messages", messages=messages)

# 反序列化消息
messages = exe.run("AI_deserialize_messages", jsondictstr=json_string)

# 创建聊天模型
chat_model = exe.run("AI__create_chat_model")

# 初始化AI
exe.create_interface_objects(class_name='ClipboardAI')

# 处理剪贴板AI类的功能
serialized = exe.run("ClipboardAI_serialize_messages", messages=messages)
user_input = exe.run("ClipboardAI_multiline_input")
updated_messages = exe.run("ClipboardAI_next", 
                            messages=messages, 
                            prompt=prompt, 
                            step_name=curr_fn())

# 序列化
json_string_direct_call = exe.run("serialize_messages", messages=messages)  # 修复为exe.run调用

# 将最后的消息与其他内容存储或处理
# memory.log(...)  # 这里需要具体实现
