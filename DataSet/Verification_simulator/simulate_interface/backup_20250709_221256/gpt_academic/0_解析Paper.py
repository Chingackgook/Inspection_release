import numpy as np
import sys
import os

import test
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.gpt_academic import ENV_DIR
from Inspection.adapters.custom_adapters.gpt_academic import *
exe = Executor('gpt_academic', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

from toolbox import update_ui
from toolbox import CatchException, report_exception
from toolbox import write_history_to_file, promote_file_to_downloadzone
from crazy_functions.crazy_utils import request_gpt_model_in_new_thread_with_ui_alive
from toolbox import ChatBotWithCookies, load_chat_cookies

def 解析Paper(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt):
    import time, glob, os
    for index, fp in enumerate(file_manifest):
        with open(fp, 'r', encoding='utf-8', errors='replace') as f:
            file_content = f.read()

        prefix = "接下来请你逐文件分析下面的论文文件，概括其内容" if index==0 else ""
        i_say = prefix + f'请对下面的文章片段用中文做一个概述，文件名是{os.path.relpath(fp, project_folder)}，文章内容是 `{file_content}`'
        i_say_show_user = prefix + f'[{index+1}/{len(file_manifest)}] 请对下面的文章片段做一个概述: {os.path.abspath(fp)}'
        chatbot.append((i_say_show_user, "[Local Message] waiting gpt response."))
        yield from exe.run("update_ui", chatbot=chatbot, history=history)  # 刷新界面
        msg = '正常'
        
        # 替换为 exe.run
        gpt_say = yield from exe.run("request_gpt_model_in_new_thread_with_ui_alive", 
                                      inputs=i_say, 
                                      inputs_show_user=i_say_show_user, 
                                      llm_kwargs=llm_kwargs, 
                                      chatbot=chatbot, 
                                      history=[], 
                                      sys_prompt=system_prompt)  # 带超时倒计时
        
        chatbot[-1] = (i_say_show_user, gpt_say)
        history.append(i_say_show_user); history.append(gpt_say)
        yield from exe.run("update_ui", chatbot=chatbot, history=history, msg=msg)  # 刷新界面
        time.sleep(2)

    all_file = ', '.join([os.path.relpath(fp, project_folder) for index, fp in enumerate(file_manifest)])
    i_say = f'根据以上你自己的分析，对全文进行概括，用学术性语言写一段中文摘要，然后再写一段英文摘要（包括{all_file}）。'
    chatbot.append((i_say, "[Local Message] waiting gpt response."))
    yield from exe.run("update_ui", chatbot=chatbot, history=history)  # 刷新界面

    msg = '正常'
    # ** gpt request **
    # 替换为 exe.run
    gpt_say = yield from exe.run("request_gpt_model_in_new_thread_with_ui_alive", 
                                  inputs=i_say, 
                                  inputs_show_user=i_say, 
                                  llm_kwargs=llm_kwargs, 
                                  chatbot=chatbot, 
                                  history=history, 
                                  sys_prompt=system_prompt)  # 带超时倒计时

    chatbot[-1] = (i_say, gpt_say)
    history.append(i_say); history.append(gpt_say)
    yield from exe.run("update_ui", chatbot=chatbot, history=history, msg=msg)  # 刷新界面
    res = write_history_to_file(history)
    promote_file_to_downloadzone(res, chatbot=chatbot)
    chatbot.append(("完成了吗？", res))
    yield from exe.run("update_ui", chatbot=chatbot, history=history, msg=msg)  # 刷新界面



test_dir = ENV_DIR+"/test_dir"
file1 = os.path.join(ENV_DIR, "test_paper1.txt")
file2 = os.path.join(ENV_DIR, "test_paper2.txt")
with open(file1, "w", encoding="utf-8") as f:
    f.write("本论文主要研究人工智能在医疗领域的应用，包括诊断和治疗建议。")
with open(file2, "w", encoding="utf-8") as f:
    f.write("本文提出了一种新的深度学习模型，用于图像识别任务，并在多个数据集上取得了优异的成绩。")
cookies = load_chat_cookies()
llm_kwargs = {
    "api_key": cookies["api_key"],
    "llm_model": "gpt-4o-mini",
    "top_p": 1.0,
    "max_length": None,
    "temperature": 1.0,
}
# 构造参数
file_manifest = [file1, file2]
project_folder = test_dir
plugin_kwargs = {}  # 可根据需要填写
chatbot = ChatBotWithCookies(llm_kwargs)
history = []
system_prompt = "你是一名学术论文分析助手，请根据输入内容进行专业分析。"

# 运行解析Paper函数
gen = 解析Paper(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt)
try:
    while True:
        next(gen)
except StopIteration:
    print("测试结束。")
