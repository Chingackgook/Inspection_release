import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.CosyVoice import ENV_DIR
from Inspection.adapters.custom_adapters.CosyVoice import *

# 创建Executor对象
exe = Executor('CosyVoice', 'simulation')


# python script.py -maodel_dir name 
# 加载模型
exe.create_interface_objects(model_dir=os.path.join(ENV_DIR, 'pretrained_models/CosyVoice2-0.5B'), load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)


# half 

FILE_RECORD_PATH = exe.now_record_path

import sys
sys.path.append(os.path.join(ENV_DIR, 'third_party/Matcha-TTS'))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 零样本使用示例
prompt_speech_16k = load_wav(os.path.join(ENV_DIR, 'asset/zero_shot_prompt.wav'), 16000)

# 替换为 exe.run 调用
for i, j in enumerate(exe.run("inference_zero_shot", 
                               tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                               prompt_text='希望你以后能够做的比我还好呦。',
                               prompt_speech_16k=prompt_speech_16k, 
                               stream=False)):
    torchaudio.save(os.path.join(FILE_RECORD_PATH, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], exe.cosyvoice.sample_rate)

# 保存零样本说话人以备后用
assert exe.run("add_zero_shot_spk", 
                prompt_text='希望你以后能够做的比我还好呦。',
                prompt_speech_16k=prompt_speech_16k, 
                zero_shot_spk_id='my_zero_shot_spk') is True

exe.run("save_spkinfo")

# 指令使用示例
for i, j in enumerate(exe.run("inference_instruct2", 
                               tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
                               instruct_text='用四川话说这句话',
                               prompt_speech_16k=prompt_speech_16k, 
                               stream=False)):
    torchaudio.save(os.path.join(FILE_RECORD_PATH, 'instruct_{}.wav'.format(i)), j['tts_speech'], exe.cosyvoice.sample_rate)

# 其他函数调用示例
for i, j in enumerate(exe.run("inference_cross_lingual", 
                               tts_text='在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。',
                               prompt_speech_16k=prompt_speech_16k, 
                               stream=False)):
    torchaudio.save(os.path.join(FILE_RECORD_PATH, 'fine_grained_control_{}.wav'.format(i)), j['tts_speech'], exe.cosyvoice.sample_rate)

# 语音转换示例
source_speech_16k = load_wav(os.path.join(ENV_DIR, 'asset/source_speech.wav'), 16000)
for i, j in enumerate(exe.run("inference_vc", 
                               source_speech_16k=source_speech_16k, 
                               prompt_speech_16k=prompt_speech_16k, 
                               stream=False)):
    torchaudio.save(os.path.join(FILE_RECORD_PATH, 'vc_{}.wav'.format(i)), j['tts_speech'], exe.cosyvoice.sample_rate)
