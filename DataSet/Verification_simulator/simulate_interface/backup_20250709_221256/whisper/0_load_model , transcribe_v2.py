from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.whisper import ENV_DIR
from Inspection.adapters.custom_adapters.whisper import *
exe = Executor('whisper', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
AUDIO_FILE = 'audio_file.wav'  # 输入音频文件路径
MODEL_NAME = 'medium'           # 模型名称
OUTPUT_DIR = '.'                # 输出目录
# end

# 导入原有的包
import argparse
import os
import traceback
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import tqdm
from audio import FRAMES_PER_SECOND
from audio import HOP_LENGTH
from audio import N_FRAMES
from audio import N_SAMPLES
from audio import SAMPLE_RATE
from audio import log_mel_spectrogram
from audio import pad_or_trim
from decoding import DecodingOptions
from decoding import DecodingResult
from timing import add_word_timestamps
from tokenizer import LANGUAGES
from tokenizer import TO_LANGUAGE_CODE
from tokenizer import get_tokenizer
from utils import exact_div
from utils import format_timestamp
from utils import get_end
from utils import get_writer
from utils import make_safe
from utils import optional_float
from utils import optional_int
from utils import str2bool
from model import Whisper

def valid_model_name(name):
    available_models = ['tiny', 'base', 'small', 'medium', 'large', 'large.en']
    if name in available_models or os.path.exists(name):
        return name
    raise ValueError(f'model should be one of {available_models} or path to a model checkpoint')

def cli():
    args = {
        'audio': [os.path.join(ENV_DIR, AUDIO_FILE)],  # 使用全局变量ENV_DIR
        'model': MODEL_NAME,                             # 使用全局变量MODEL_NAME
        'model_dir': None,
        'device': 'cpu',
        'output_dir': FILE_RECORD_PATH,                 # 使用全局变量FILE_RECORD_PATH
        'output_format': 'all',
        'verbose': True,
        'task': 'transcribe',
        'language': None,
        'temperature': 0,
        'best_of': 5,
        'beam_size': 5,
        'patience': None,
        'length_penalty': None,
        'suppress_tokens': '-1',
        'initial_prompt': None,
        'carry_initial_prompt': False,
        'condition_on_previous_text': True,
        'fp16': True,
        'temperature_increment_on_fallback': 0.2,
        'compression_ratio_threshold': 2.4,
        'logprob_threshold': -1.0,
        'no_speech_threshold': 0.6,
        'word_timestamps': False,
        'prepend_punctuations': '"\'“¿([{-',
        'append_punctuations': '"\'.。,，!！?？:：”)]}、',
        'highlight_words': False,
        'max_line_width': None,
        'max_line_count': None,
        'max_words_per_line': None,
        'threads': 0,
        'clip_timestamps': '0',
        'hallucination_silence_threshold': None
    }
    
    model_name: str = args.pop('model')
    model_dir: str = args.pop('model_dir')
    output_dir: str = args.pop('output_dir')
    output_format: str = args.pop('output_format')
    device: str = args.pop('device')
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name.endswith('.en') and args['language'] not in {'en', 'English'}:
        if args['language'] is not None:
            warnings.warn(f"{model_name} is an English-only model but received '{args['language']}'; using English instead.")
        args['language'] = 'en'
    
    temperature = args.pop('temperature')
    if (increment := args.pop('temperature_increment_on_fallback')) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-06, increment))
    else:
        temperature = [temperature]
    
    if (threads := args.pop('threads')) > 0:
        torch.set_num_threads(threads)
    
    from model import load_model
    model = load_model(model_name, device=device, download_root=model_dir)
    writer = get_writer(output_format, output_dir)
    
    word_options = ['highlight_words', 'max_line_count', 'max_line_width', 'max_words_per_line']
    if not args['word_timestamps']:
        for option in word_options:
            if args[option]:
                raise ValueError(f'--{option} requires --word_timestamps True')
    
    if args['max_line_count'] and (not args['max_line_width']):
        warnings.warn('--max_line_count has no effect without --max_line_width')
    
    if args['max_words_per_line'] and args['max_line_width']:
        warnings.warn('--max_words_per_line has no effect with --max_line_width')
    
    writer_args = {arg: args.pop(arg) for arg in word_options}
    
    for audio_path in args.pop('audio'):
        try:
            result = exe.run('transcribe', model=model, audio=audio_path, temperature=temperature, **args)
            writer(result, audio_path, **writer_args)
        except Exception as e:
            traceback.print_exc()
            print(f'Skipping {audio_path} due to {type(e).__name__}: {str(e)}')

cli()
