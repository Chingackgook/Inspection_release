import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.ChatGLM3 import ENV_DIR
from Inspection.adapters.custom_adapters.ChatGLM3 import *
exe = Executor('ChatGLM3', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
import platform
from transformers import AutoTokenizer
from transformers import AutoModel
import os
import platform
MODEL_PATH = os.path.join(ENV_DIR, 'chatglm3-6b')
TOKENIZER_PATH = MODEL_PATH
exe.create_interface_objects(model_path=MODEL_PATH)
exe.create_interface_objects(model_path=TOKENIZER_PATH)
tokenizer = exe.adapter.auto_tokenizer
model = exe.adapter.auto_model.eval()
os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False
welcome_prompt = '欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序'

def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f'\n\n用户：{query}'
        prompt += f'\n\nChatGLM3-6B：{response}'
    return prompt
past_key_values, history = (None, [])

print(welcome_prompt)
user_inputs = ['你好', '你是谁？', 'stop']
for query in user_inputs:
    if query.strip() == 'stop':
        break
    print('\nChatGLM：', end='')
    current_length = 0
    for response, history, past_key_values in model.stream_chat(exe.adapter.auto_tokenizer, query, history=history, top_p=1, temperature=0.01, past_key_values=past_key_values, return_past_key_values=True):
        if stop_stream:
            stop_stream = False
            break
        else:
            print(response[current_length:], end='', flush=True)
            current_length = len(response)
    print('')