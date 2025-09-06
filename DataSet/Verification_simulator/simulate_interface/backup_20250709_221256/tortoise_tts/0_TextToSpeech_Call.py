import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.tortoise_tts import ENV_DIR
from Inspection.adapters.custom_adapters.tortoise_tts import *
exe = Executor('tortoise_tts', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import argparse
import os
import torch
import torchaudio
from api import TextToSpeech
from api import MODELS_DIR
from utils.audio import load_voices
text = 'The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.'
voice = 'random'
preset = 'fast'
use_deepspeed = False
kv_cache = True
half = True
output_path = FILE_RECORD_PATH
model_dir = ENV_DIR
candidates = 3
seed = None
produce_debug_state = True
cvvp_amount = 0.0
if torch.backends.mps.is_available():
    use_deepspeed = False
os.makedirs(output_path, exist_ok=True)
exe.create_interface_objects(models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
selected_voices = voice.split(',')
for (k, selected_voice) in enumerate(selected_voices):
    if '&' in selected_voice:
        voice_sel = selected_voice.split('&')
    else:
        voice_sel = [selected_voice]
    (voice_samples, conditioning_latents) = load_voices(voice_sel)
    conditioning_latents = exe.run('get_conditioning_latents', voice_samples=voice_samples, return_mels=False)
    (gen, dbg_state) = exe.run('tts_with_preset', text=text, preset=preset, voice_samples=voice_samples, conditioning_latents=conditioning_latents, k=candidates, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount)
    if isinstance(gen, list):
        for (j, g) in enumerate(gen):
            torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
    else:
        torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)
    if produce_debug_state:
        os.makedirs('debug_states', exist_ok=True)
        torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')