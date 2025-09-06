
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.bark import ENV_DIR
from Inspection.adapters.custom_adapters.bark import *
exe = Executor('bark','simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
from scipy.io.wavfile import write as write_wav

# 假设 exe 对象已经被初始化
exe.create_interface_objects()  # 加载模型

# 基础参数输入
input_text = "Hello, world!"
output_filename = "bark_generation.wav"
output_dir = FILE_RECORD_PATH  # 替换为全局变量 FILE_RECORD_PATH
history_prompt = None
text_temp = 0.7
waveform_temp = 0.7
silent = False
output_full = False

try:
    os.makedirs(output_dir, exist_ok=True)
    
    # 替换 generate_audio 调用
    from Inspection.adapters.custom_adapters.bark import generate_audio
    
    exe.run("generate_audio", 
        text=input_text,
        history_prompt=history_prompt,
        text_temp=text_temp,
        waveform_temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    
    # 从 exe.adapter.result 获取生成的音频数据
    generated_audio = exe.adapter.result.except_data
    
    output_file_path = os.path.join(output_dir, output_filename)
    write_wav(output_file_path, SAMPLE_RATE, generated_audio)
    print(f"Done! Output audio file is saved at: '{output_file_path}'")
    
except Exception as e:
    print(f"Oops, an error occurred: {e}")
