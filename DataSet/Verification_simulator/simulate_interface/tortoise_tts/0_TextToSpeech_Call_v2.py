from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.tortoise_tts import *
from Inspection.adapters.custom_adapters.tortoise_tts import ENV_DIR
exe = Executor('tortoise_tts', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/tortoise-tts/tortoise/do_tts.py'
import argparse
import os
import torch
import torchaudio
from api import TextToSpeech
from api import MODELS_DIR
from utils.audio import load_voices
# end

import os
import torch
import torchaudio
from api import TextToSpeech, MODELS_DIR
from utils.audio import load_voices

def run_tts():
    text = 'The expressiveness of autoregressive transformers is literally nuts! I absolutely adore them.'
    voice = 'random'
    preset = 'fast'
    use_deepspeed = False
    kv_cache = True
    half = True
    output_path = FILE_RECORD_PATH  # Use the global variable for output path
    # add
    model_dir = ENV_DIR
    # end add
    # Origin 
    # model_dir = MODELS_DIR
    candidates = 3
    seed = None
    produce_debug_state = True
    cvvp_amount = 0.0
    if torch.backends.mps.is_available():
        use_deepspeed = False
    os.makedirs(output_path, exist_ok=True)
    tts = exe.create_interface_objects(interface_class_name='TextToSpeech', models_dir=model_dir, use_deepspeed=use_deepspeed, kv_cache=kv_cache, half=half)
    selected_voices = voice.split(',')
    for (k, selected_voice) in enumerate(selected_voices):
        if '&' in selected_voice:
            voice_sel = selected_voice.split('&')
        else:
            voice_sel = [selected_voice]
        (voice_samples, conditioning_latents) = load_voices(voice_sel)
        (gen, dbg_state) = exe.run('tts_with_preset', text=text, preset=preset, voice_samples=voice_samples, conditioning_latents=conditioning_latents, k=candidates, use_deterministic_seed=seed, return_deterministic_state=True, cvvp_amount=cvvp_amount)
        
        # Save generated audio files
        if isinstance(gen, list):
            for (j, g) in enumerate(gen):
                torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}_{j}.wav'), g.squeeze(0).cpu(), 24000)
        else:
            torchaudio.save(os.path.join(output_path, f'{selected_voice}_{k}.wav'), gen.squeeze(0).cpu(), 24000)
        
        # Save debug state if required
        if produce_debug_state:
            os.makedirs('debug_states', exist_ok=True)
            torch.save(dbg_state, f'debug_states/do_tts_debug_{selected_voice}.pth')

# Run the TTS function directly
run_tts()
