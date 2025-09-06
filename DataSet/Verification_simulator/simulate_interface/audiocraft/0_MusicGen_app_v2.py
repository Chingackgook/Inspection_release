from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.audiocraft import *
exe = Executor('audiocraft','simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package
import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings
from einops import rearrange
import torch
import gradio as gr
from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion

MODEL = None
SPACE_ID = 'facebook/MusicGen'
IS_BATCHED = 'facebook/MusicGen' in SPACE_ID or 'musicgen-internal/musicgen_dev' in SPACE_ID
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
_old_call = sp.call

def _call_nostderr(*args, **kwargs):
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)

sp.call = _call_nostderr
pool = ProcessPoolExecutor(4)
pool.__enter__()

def interrupt():
    global INTERRUPTING
    INTERRUPTING = True

class FileCleaner:

    def __init__(self, file_lifetime: float=3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break

file_cleaner = FileCleaner()

def make_waveform(*args, **kwargs):
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print('Make a video took', time.time() - be)
        return out

def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print('Loading model', version)
    if MODEL is None or MODEL.name != version:
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None
        MODEL = exe.create_interface_objects(interface_class_name='MusicGen', name=version)

def load_diffusion():
    global MBD
    if MBD is None:
        print('loading MBD')
        MBD = MultiBandDiffusion.get_mbd_musicgen()

def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    exe.run('set_generation_params', duration=duration, **gen_kwargs)
    print('new batch', len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = (melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t())
            if melody.dim() == 1:
                melody = None
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)
    try:
        if any((m is not None for m in processed_melodies)):
            outputs = exe.run('generate_with_chroma', descriptions=texts, melody_wavs=processed_melodies, melody_sample_rate=target_sr, progress=progress, return_tokens=USE_DIFFUSION)
        else:
            outputs = exe.run('generate', descriptions=texts, progress=progress, return_tokens=USE_DIFFUSION)
    except RuntimeError as e:
        raise Exception('Error while generating ' + e.args[0])
    
    if USE_DIFFUSION:
        tokens = outputs[1]
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            left, right = MODEL.compression_model.get_left_right_codes(tokens)
            tokens = torch.cat([left, right])
        outputs_diffusion = MBD.tokens_to_wav(tokens)
        if isinstance(MODEL.compression_model, InterleaveStereoCompressionModel):
            assert outputs_diffusion.shape[1] == 1
            outputs_diffusion = rearrange(outputs_diffusion, '(s b) c t -> b (s c) t', s=2)
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        # Use FILE_RECORD_PATH for output file paths
        output_file_path = os.path.join(FILE_RECORD_PATH, f'temp_output_{time.time()}.wav')
        with open(output_file_path, 'wb') as file:
            audio_write(file.name, output, MODEL.sample_rate, strategy='loudness', loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, output_file_path))
            out_wavs.append(output_file_path)
            file_cleaner.add(output_file_path)
    
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    
    print('batch finished', len(texts), time.time() - be)
    print('Tempfiles currently stored: ', len(file_cleaner.files))
    return (out_videos, out_wavs)

def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/musicgen-stereo-melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return res

# Parts that may need manual modification:
model = 'facebook/musicgen-stereo-melody'
model_path = ''
decoder = 'Default'
text = 'An 80s driving pop song with heavy drums and synth pads in the background'
melody = None
duration = 10
topk = 250
topp = 0
temperature = 1.0
cfg_coef = 3.0
USE_DIFFUSION = False
# end

load_model(model)
videos, wavs = predict_batched([text], [melody])
print('Generated Videos:', videos)
print('Generated WAVs:', wavs)