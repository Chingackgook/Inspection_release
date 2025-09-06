from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.CosyVoice import *
exe = Executor('CosyVoice','simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/CosyVoice/generate.py'
# Import the existing package
import sys
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
# end

import os

# Check for required model and audio files
if not os.path.exists('pretrained_models/CosyVoice2-0.5B'):
    raise FileNotFoundError("Model directory 'pretrained_models/CosyVoice2-0.5B' does not exist.")
if not os.path.exists('./asset/zero_shot_prompt.wav'):
    raise FileNotFoundError("Prompt audio file './asset/zero_shot_prompt.wav' does not exist.")

sys.path.append('third_party/Matcha-TTS')
from cosyvoice.utils.file_utils import load_wav
import torchaudio

def main():
    cosyvoice = exe.create_interface_objects(interface_class_name='CosyVoice2', model_dir='pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

    # First inference with zero shot
    for (i, j) in enumerate(exe.run('inference_zero_shot', tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', prompt_text='希望你以后能够做的比我还好呦。', prompt_speech_16k=prompt_speech_16k, stream=False)):
        torchaudio.save(os.path.join(FILE_RECORD_PATH, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)

    # Adding zero shot speaker
    assert exe.run('add_zero_shot_spk', prompt_text='希望你以后能够做的比我还好呦。', prompt_speech_16k=prompt_speech_16k, zero_shot_spk_id='my_zero_shot_spk') is True

    # Second inference with the added speaker
    for (i, j) in enumerate(exe.run('inference_zero_shot', tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', prompt_text='', prompt_speech_16k='', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
        torchaudio.save(os.path.join(FILE_RECORD_PATH, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)

    # Save speaker info
    cosyvoice.save_spkinfo()

    # Fine-grained control inference
    for (i, j) in enumerate(exe.run('inference_cross_lingual', tts_text='在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。', prompt_speech_16k=prompt_speech_16k, stream=False)):
        torchaudio.save(os.path.join(FILE_RECORD_PATH, 'fine_grained_control_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)

    # Instruction inference
    for (i, j) in enumerate(exe.run('inference_instruct2', tts_text='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', instruct_text='用四川话说这句话', prompt_speech_16k=prompt_speech_16k, stream=False)):
        torchaudio.save(os.path.join(FILE_RECORD_PATH, 'instruct_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)

    # Text generator for zero shot inference
    def text_generator():
        yield '收到好友从远方寄来的生日礼物，'
        yield '那份意外的惊喜与深深的祝福'
        yield '让我心中充满了甜蜜的快乐，'
        yield '笑容如花儿般绽放。'

    # Final inference with text generator
    for (i, j) in enumerate(exe.run('inference_zero_shot', tts_text=text_generator(), prompt_text='希望你以后能够做的比我还好呦。', prompt_speech_16k=prompt_speech_16k, stream=False)):
        torchaudio.save(os.path.join(FILE_RECORD_PATH, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)

# Directly run the main logic
main()
