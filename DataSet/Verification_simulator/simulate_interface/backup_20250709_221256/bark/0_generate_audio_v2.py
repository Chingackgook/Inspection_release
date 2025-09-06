from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.bark import ENV_DIR
from Inspection.adapters.custom_adapters.bark import *
exe = Executor('bark', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
input_text = 'Hello, world!'  # 输入文本
output_filename = 'bark_generation.wav'  # 输出文件名
output_dir = '.'  # 输出目录
text_temp = 0.7  # 文本温度
waveform_temp = 0.7  # 波形温度
silent = False  # 是否静音
output_full = False  # 是否输出完整音频
# end

import os
from scipy.io.wavfile import write as write_wav
from bark.api import generate_audio
from bark.generation import SAMPLE_RATE

def generate_audio_from_text():
    """Generate audio from input text."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        generated_audio = exe.run('generate_audio', text=input_text, history_prompt=None, text_temp=text_temp, waveform_temp=waveform_temp, silent=silent, output_full=output_full)
        output_file_path = os.path.join(FILE_RECORD_PATH, output_filename)  # 使用 FILE_RECORD_PATH 作为输出路径
        write_wav(output_file_path, SAMPLE_RATE, generated_audio)
        print(f"Done! Output audio file is saved at: '{output_file_path}'")
    except Exception as e:
        print(f'Oops, an error occurred: {e}')

# 直接运行主逻辑
generate_audio_from_text()
