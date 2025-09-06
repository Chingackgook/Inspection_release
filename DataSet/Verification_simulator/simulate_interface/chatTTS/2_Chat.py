
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
import torch
import ChatTTS

# 设置环境变量
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 获取当前目录
now_dir = os.getcwd()
sys.path.append(now_dir)

# 定义路径常量


# 创建 exe 实例

# 加载模型
exe.create_interface_objects(source="local", custom_path=ENV_DIR)  # 使用 load_model 方法加载模型

# 尝试注册文本正规化工具

# 随机选择说话者
rand_spk = exe.run("sample_random_speaker")

# 定义文本
text = ["What is [uv_break]your favorite english food?[laugh][lbreak]"]

# 初始化失败标志
fail = False

# 使用 exe.run 替换 infer 方法
refined_text = exe.run(
    "infer",
    text=text,
    refine_text_only=True,
    params_refine_text=ChatTTS.Chat.RefineTextParams(
        prompt="[oral_2][laugh_0][break_6]",
        manual_seed=12345,
    ),
    split_text=False,
)

# 检查精炼后的文本
if (
    refined_text[0]
    != "what is [uv_break] your favorite english [uv_break] food [laugh] like [lbreak]"
):
    fail = True

# 定义推理参数
params = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,  # add sampled speaker
    temperature=0.3,  # using custom temperature
    top_P=0.7,  # top P decode
    top_K=20,  # top K decode
)

# 编码文本
input_ids, attention_mask, text_mask = exe.adapter.chat.tokenizer.encode(  # 使用 exe.adapter.chat
    exe.adapter.chat.speaker.decorate_code_prompts(  # 使用 exe.adapter.chat
        text,
        params.prompt,
        params.txt_smp,
        params.spk_emb,
    ),
    exe.adapter.chat.config.gpt.num_vq,  # 使用 exe.adapter.chat
    prompt=(
        exe.adapter.chat.speaker.decode_prompt(params.spk_smp)  # 使用 exe.adapter.chat
        if params.spk_smp is not None
        else None
    ),
    device=exe.adapter.chat.device_gpt,  # 使用 exe.adapter.chat
)

# 使用 exe.run 替换推理过程
with torch.inference_mode():
    start_idx, end_idx = 0, torch.zeros(
        input_ids.shape[0], device=input_ids.device, dtype=torch.long
    ).fill_(input_ids.shape[1])

    recoded_text = exe.run(
        "infer",
        text=input_ids,
        stream=False,
        lang=None,
        skip_refine_text=False,
        refine_text_only=False,
        use_decoder=True,
        do_text_normalization=True,
        do_homophone_replacement=True,
        split_text=True,
        max_split_batch=4,
        params_infer_code=params,
    )

# 检查重编码的文本
if (
    recoded_text[0]
    != "[Stts] [spk_emb] [speed_5] what is [uv_break] your favorite english food? [laugh] [lbreak] [Ptts]"
):
    fail = True

# 如果有失败，退出
if fail:
    print("Test failed")

# 卸载模型
exe.run("unload")
