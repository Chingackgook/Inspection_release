
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.chatTTS import ENV_DIR
from Inspection.adapters.custom_adapters.chatTTS import *
exe = Executor('chatTTS','simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
import sys
import logging
import re
import ChatTTS

# 设置环境变量
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 获取当前目录
now_dir = os.getcwd()
sys.path.append(now_dir)

# 初始化日志记录器

# 初始化 exe 类（假设 exe 是在某处定义的）

# 加载模型（调用 load_model 方法）
exe.create_interface_objects(compile=False, source="huggingface")  # Set to True for better performance

# 定义文本
texts = [
    "总结一下，AI Agent是大模型功能的扩展，让AI更接近于通用人工智能，也就是我们常说的AGI。",
    "你真是太聪明啦。",
]

# 下载模型
exe.run("download_models", source="huggingface", force_redownload=False)

# 检查是否加载
if not exe.run("has_loaded", use_decoder=False):
    print("模型加载失败")

# 推理文本
refined = exe.run(
    "infer",
    text=texts,
    stream=False,
    lang=None,
    skip_refine_text=False,
    refine_text_only=True,
    use_decoder=True,
    do_text_normalization=True,
    do_homophone_replacement=True,
    split_text=False,
    max_split_batch=4,
    params_refine_text=ChatTTS.Chat.RefineTextParams(show_tqdm=False),
)

# 定义去除标签的函数
trimre = re.compile("\\[[\w_]+\\]")

def trim_tags(txt: str) -> str:
    global trimre
    return trimre.sub("", txt)

# 检查推理结果
fail = False
for i, t in enumerate(refined):
    if len(trim_tags(t)) > 4 * len(texts[i]):
        fail = True

if fail:
    print("推理结果不符合预期")

# 卸载模型
exe.run("unload")

# 随机选择说话者
speaker = exe.run("sample_random_speaker")

# 假设我们有一个 wav_data 变量用于编码，确保这里有实际的 wav 数据
wav_data = ...  # 这里需要提供实际的 wav 数据
encoded_speaker = exe.run("sample_audio_speaker", wav=wav_data)

# 中断当前上下文
exe.run("interrupt")
