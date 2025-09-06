import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.audiocraft import ENV_DIR
from Inspection.adapters.custom_adapters.audiocraft import *
import torch

exe = Executor('audiocraft', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 使用exe对象加载模型
exe.create_interface_objects(name='facebook/musicgen-melody')  # 确保模型被加载

# 设置生成参数
duration = 30.0
temperature = 0.8
top_k = 250
top_p = 0.0
cfg_coef = 3.0

# 设置生成参数
exe.run("set_generation_params", duration=duration, temperature=temperature, top_k=top_k, top_p=top_p, cfg_coef=cfg_coef)

# 生成音乐
descriptions = ["A calm and soothing melody", "An upbeat and lively tune"]
melody_wavs = [torch.randn(1, 2, 44100), None]  # 示例旋律波形
melody_sample_rate = 44100

# 生成音频
generated_audio = exe.run("generate_with_chroma", descriptions=descriptions, melody_wavs=melody_wavs, melody_sample_rate=melody_sample_rate, progress=False, return_tokens=False)

# 处理生成的音频（例如保存或播放）
# 这里可以添加保存或播放音频的代码，使用FILE_RECORD_PATH作为输出路径
