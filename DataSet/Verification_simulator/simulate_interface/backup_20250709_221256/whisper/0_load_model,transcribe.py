
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.whisper import ENV_DIR
exe = Executor('whisper', 'simulation')
exe.set_record_function(["transcribe"])
FILE_RECORD_PATH = exe.now_record_path

import torch
import numpy as np
from whisper import available_models, load_model, transcribe ,log_mel_spectrogram ,load_audio
from whisper.decoding import DecodingOptions
from whisper.model import Whisper

# 模拟音频波形
audio_waveform = load_audio(os.path.join(ENV_DIR, "audio_file.wav"))  # 假设音频文件路径

# 加载模型
model_name = "tiny"  # 假设模型名称
device = "cuda" if torch.cuda.is_available() else "cpu"
model = exe.create_interface_objects(name=model_name, device=device)  # 正确调用load_model


# 设置对齐头
alignment_heads = bytes([1, 0, 1, 0])  # 假设的布尔数组示例
exe.run("set_alignment_heads", alignment_heads=alignment_heads)  # 正确调用，set_alignment_heads已实现

# 生成梅尔频谱段
mel_segment = log_mel_spectrogram(audio_waveform)  # 从音频波形生成梅尔频谱段

# 检测语言
detected_language = exe.run("detect_language", mel_segment=mel_segment)  # 正确调用，detect_language已实现

# 解码选项
decoding_options = DecodingOptions(temperature=0.0, language=detected_language, task="transcribe")  # detected_language应为单个语言字符串

# 解码梅尔频谱段
decoding_result = exe.run("decode", segment=mel_segment, options=decoding_options)  # 正确调用，decode已实现

# 转录音频
result = exe.run("transcribe", model=model, audio=audio_waveform, temperature=0.0, verbose=True)  # 正确调用，transcribe已实现

# 输出结果
print(result)
