from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.F5_TTS import *
exe = Executor('F5_TTS', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/F5-TTS/src/f5_tts/infer/infer_cli.py'
import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import cfg_strength
from f5_tts.infer.utils_infer import cross_fade_duration
from f5_tts.infer.utils_infer import device
from f5_tts.infer.utils_infer import fix_duration
from f5_tts.infer.utils_infer import infer_process
from f5_tts.infer.utils_infer import load_model
from f5_tts.infer.utils_infer import load_vocoder
from f5_tts.infer.utils_infer import mel_spec_type
from f5_tts.infer.utils_infer import nfe_step
from f5_tts.infer.utils_infer import preprocess_ref_audio_text
from f5_tts.infer.utils_infer import remove_silence_for_generated_wav
from f5_tts.infer.utils_infer import speed
from f5_tts.infer.utils_infer import sway_sampling_coef
from f5_tts.infer.utils_infer import target_rms
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from f5_tts.infer.utils_infer import cfg_strength, cross_fade_duration, device, fix_duration, infer_process, load_model, load_vocoder, mel_spec_type, nfe_step, preprocess_ref_audio_text, remove_silence_for_generated_wav, speed, sway_sampling_coef, target_rms
args = {'config': os.path.join(files("f5_tts").joinpath("infer/examples/basic"), "basic.toml"), 'model': 'F5TTS_v1_Base', 'model_cfg': '', 'ckpt_file': '', 'vocab_file': '', 'ref_audio': RESOURCES_PATH + 'audios/test_audio.wav', 'ref_text': 'The content, subtitle or transcription of reference audio.', 'gen_text': 'Some text you want TTS model generate for you.', 'gen_file': '', 'output_dir': 'output', 'output_file': f"infer_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav", 'save_chunk': False, 'remove_silence': False, 'load_vocoder_from_local': False, 'vocoder_name': mel_spec_type, 'target_rms': target_rms, 'cross_fade_duration': cross_fade_duration, 'nfe_step': nfe_step, 'cfg_strength': cfg_strength, 'sway_sampling_coef': sway_sampling_coef, 'speed': speed, 'fix_duration': fix_duration, 'device': device}
config = tomli.load(open(args['config'], 'rb'))
model = args['model'] or config.get('model', 'F5TTS_v1_Base')
ckpt_file = args['ckpt_file'] or config.get('ckpt_file', '')
vocab_file = args['vocab_file'] or config.get('vocab_file', '')
ref_audio = RESOURCES_PATH + 'audios/test_audio.wav'
ref_text = args['ref_text'] if args['ref_text'] is not None else config.get('ref_text', 'Some call me nature, others call me mother nature.')
gen_text = args['gen_text'] or config.get('gen_text', 'Here we generate something just for test.')
gen_file = args['gen_file'] or config.get('gen_file', '')
output_dir = args['output_dir'] or config.get('output_dir', 'tests')
output_file = args['output_file'] or config.get('output_file', f"infer_cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
save_chunk = args['save_chunk'] or config.get('save_chunk', False)
remove_silence = args['remove_silence'] or config.get('remove_silence', False)
load_vocoder_from_local = args['load_vocoder_from_local'] or config.get('load_vocoder_from_local', False)
vocoder_name = args['vocoder_name'] or config.get('vocoder_name', mel_spec_type)
target_rms = args['target_rms'] or config.get('target_rms', target_rms)
cross_fade_duration = args['cross_fade_duration'] or config.get('cross_fade_duration', cross_fade_duration)
nfe_step = args['nfe_step'] or config.get('nfe_step', nfe_step)
cfg_strength = args['cfg_strength'] or config.get('cfg_strength', cfg_strength)
sway_sampling_coef = args['sway_sampling_coef'] or config.get('sway_sampling_coef', sway_sampling_coef)
speed = args['speed'] or config.get('speed', speed)
fix_duration = args['fix_duration'] or config.get('fix_duration', fix_duration)
device = args['device'] or config.get('device', device)
if 'infer/examples/' in ref_audio:
    ref_audio = str(files('f5_tts').joinpath(f'{ref_audio}'))
if 'infer/examples/' in gen_file:
    gen_file = str(files('f5_tts').joinpath(f'{gen_file}'))
if 'voices' in config:
    for voice in config['voices']:
        voice_ref_audio = config['voices'][voice]['ref_audio']
        if 'infer/examples/' in voice_ref_audio:
            config['voices'][voice]['ref_audio'] = str(files('f5_tts').joinpath(f'{voice_ref_audio}'))
if gen_file:
    gen_text = codecs.open(gen_file, 'r', 'utf-8').read()
wave_path = Path(FILE_RECORD_PATH) / output_file
if save_chunk:
    output_chunk_dir = os.path.join(FILE_RECORD_PATH, f'{Path(output_file).stem}_chunks')
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)
if vocoder_name == 'vocos':
    vocoder_local_path = '../checkpoints/vocos-mel-24khz'
elif vocoder_name == 'bigvgan':
    vocoder_local_path = '../checkpoints/bigvgan_v2_24khz_100band_256x'
vocoder = exe.run('load_vocoder', vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device)
model_cfg = OmegaConf.load(args['model_cfg'] or config.get('model_cfg', str(files('f5_tts').joinpath(f'configs/{model}.yaml'))))
model_cls = get_class(f'f5_tts.model.{model_cfg.model.backbone}')
model_arc = model_cfg.model.arch
(repo_name, ckpt_step, ckpt_type) = ('F5-TTS', 1250000, 'safetensors')
if model != 'F5TTS_Base':
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type
if model == 'F5TTS_Base':
    if vocoder_name == 'vocos':
        ckpt_step = 1200000
    elif vocoder_name == 'bigvgan':
        model = 'F5TTS_Base_bigvgan'
        ckpt_type = 'pt'
elif model == 'E2TTS_Base':
    repo_name = 'E2-TTS'
    ckpt_step = 1200000
if not ckpt_file:
    ckpt_file = str(cached_path(f'hf://SWivid/{repo_name}/{model}/model_{ckpt_step}.{ckpt_type}'))
print(f'Using {model}...')
ema_model = exe.run('load_model', model_cls=model_cls, model_cfg=model_arc, ckpt_path=ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device)

def main():
    main_voice = {'ref_audio': RESOURCES_PATH + 'audios/test_audio.wav', 'ref_text': ref_text}
    if 'voices' not in config:
        voices = {'main': main_voice}
    else:
        voices = config['voices']
        voices['main'] = main_voice
    for voice in voices:
        print('Voice:', voice)
        print('ref_audio ', voices[voice]['ref_audio'])
        (voices[voice]['ref_audio'], voices[voice]['ref_text']) = exe.run('preprocess_ref_audio_text', ref_audio_orig=voices[voice]['ref_audio'], ref_text=voices[voice]['ref_text'])
        print('ref_audio_', voices[voice]['ref_audio'], '\n\n')
    generated_audio_segments = []
    reg1 = '(?=\\[\\w+\\])'
    chunks = re.split(reg1, gen_text)
    reg2 = '\\[(\\w+)\\]'
    for text in chunks:
        if not text.strip():
            continue
        match = re.match(reg2, text)
        if match:
            voice = match[1]
        else:
            print('No voice tag found, using main.')
            voice = 'main'
        if voice not in voices:
            print(f'Voice {voice} not found, using main.')
            voice = 'main'
        text = re.sub(reg2, '', text)
        ref_audio_ = voices[voice]['ref_audio']
        ref_text_ = voices[voice]['ref_text']
        gen_text_ = text.strip()
        print(f'Voice: {voice}')
        (audio_segment, final_sample_rate, spectrogram) = exe.run('infer_process', ref_audio=ref_audio_, ref_text=ref_text_, gen_text=gen_text_, model_obj=ema_model, vocoder=vocoder, mel_spec_type=vocoder_name, target_rms=target_rms, cross_fade_duration=cross_fade_duration, nfe_step=nfe_step, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef, speed=speed, fix_duration=fix_duration, device=device)
        generated_audio_segments.append(audio_segment)
        if save_chunk:
            if len(gen_text_) > 200:
                gen_text_ = gen_text_[:200] + ' ... '
            sf.write(os.path.join(output_chunk_dir, f'{len(generated_audio_segments) - 1}_{gen_text_}.wav'), audio_segment, final_sample_rate)
    if generated_audio_segments:
        final_wave = np.concatenate(generated_audio_segments)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(wave_path, 'wb') as f:
            sf.write(f.name, final_wave, final_sample_rate)
            if remove_silence:
                exe.run('remove_silence_for_generated_wav', filename=f.name)
            print(f.name)
main()