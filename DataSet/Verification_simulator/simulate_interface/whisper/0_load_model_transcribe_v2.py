from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.whisper import *
exe = Executor('whisper', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/whisper/whisper/transcribe.py'
import argparse
import os
import traceback
import warnings
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy as np
import torch
import tqdm
from whisper.audio import FRAMES_PER_SECOND
from whisper.audio import HOP_LENGTH
from whisper.audio import N_FRAMES
from whisper.audio import N_SAMPLES
from whisper.audio import SAMPLE_RATE
from whisper.audio import log_mel_spectrogram
from whisper.audio import pad_or_trim
from whisper.decoding import DecodingOptions
from whisper.decoding import DecodingResult
from whisper.timing import add_word_timestamps
from whisper.tokenizer import LANGUAGES
from whisper.tokenizer import TO_LANGUAGE_CODE
from whisper.tokenizer import get_tokenizer
from whisper.utils import exact_div
from whisper.utils import format_timestamp
from whisper.utils import get_end
from whisper.utils import get_writer
from whisper.utils import make_safe
from whisper.utils import optional_float
from whisper.utils import optional_int
from whisper.utils import str2bool
from whisper.model import Whisper
from whisper import available_models
from whisper import load_model
import os
import traceback
import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from whisper.audio import FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE, log_mel_spectrogram, pad_or_trim
from whisper.decoding import DecodingOptions, DecodingResult
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from whisper.utils import exact_div, format_timestamp, get_end, get_writer, make_safe, optional_float, optional_int, str2bool
if TYPE_CHECKING:
    from whisper.model import Whisper

def execute_transcription():
    from whisper import available_models, load_model
    audio_files = [RESOURCES_PATH + 'audios/test_audio.wav']
    model_name = 'small'
    model_dir = None
    output_dir = FILE_RECORD_PATH
    output_format = 'all'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    language = None
    temperature = 0
    threads = 0
    os.makedirs(output_dir, exist_ok=True)
    if model_name.endswith('.en') and language not in {'en', 'English'}:
        if language is not None:
            warnings.warn(f"{model_name} is an English-only model but received '{language}'; using English instead.")
        language = 'en'
    if (threads := threads) > 0:
        torch.set_num_threads(threads)
    model = exe.run('load_model', name=model_name, device=device, download_root=model_dir)
    writer = get_writer(output_format, output_dir)
    word_options = ['highlight_words', 'max_line_count', 'max_line_width', 'max_words_per_line']
    writer_args = {'highlight_words': False, 'max_line_count': None, 'max_line_width': None, 'max_words_per_line': None}
    for audio_path in audio_files:
        try:
            result = exe.run('transcribe', model=model, audio=audio_path, temperature=temperature, verbose=True, language=language)
            writer(result, audio_path, **writer_args)
        except Exception as e:
            traceback.print_exc()
            print(f'Skipping {audio_path} due to {type(e).__name__}: {str(e)}')
execute_transcription()