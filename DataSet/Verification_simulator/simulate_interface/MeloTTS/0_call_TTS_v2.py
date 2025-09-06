from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.MeloTTS import *
exe = Executor('MeloTTS', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/MeloTTS/melo/main.py'
import click
import warnings
import os
from melo.api import TTS

def main():
    text = 'Hello, this is a test for text-to-speech conversion.'
    output_path = os.path.join(FILE_RECORD_PATH, 'audio.wav')
    file = False
    language = 'EN'
    speaker = 'EN-Default'
    speed = 1.0
    device = 'auto'
    if file:
        if not os.path.exists(text):
            raise FileNotFoundError(f'Trying to load text from file due to --file/-f flag, but file not found. Remove the --file/-f flag to pass a string.')
        else:
            with open(text) as f:
                text = f.read().strip()
    if text == '':
        raise ValueError('You entered empty text or the file you passed was empty.')
    language = language.upper()
    if language == '':
        language = 'EN'
    if speaker == '':
        speaker = None
    if not language == 'EN' and speaker:
        warnings.warn('You specified a speaker but the language is English.')
    model = exe.create_interface_objects(interface_class_name='TTS', language=language, device=device)
    speaker_ids = model.hps.data.spk2id
    if language == 'EN':
        if not speaker:
            speaker = 'EN-Default'
        spkr = speaker_ids[speaker]
    else:
        spkr = speaker_ids[list(speaker_ids.keys())[0]]
    var = exe.run('tts_to_file', text=text, speaker_id=spkr, output_path=output_path, speed=speed)
main()