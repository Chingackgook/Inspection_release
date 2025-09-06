from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Spark_TTS import *
exe = Executor('Spark_TTS', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/Spark-TTS/cli/inference.py'
import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform
from cli.SparkTTS import SparkTTS

def run_tts():
    """Perform TTS inference and save the generated audio."""
    args = {'model_dir': 'pretrained_models/Spark-TTS-0.5B', 'save_dir': FILE_RECORD_PATH, 'device': 0, 'text': 'Hello, this is a test of the text-to-speech system.', 'prompt_text': None, 'prompt_speech_path': None, 'gender': 'female', 'pitch': 'moderate', 'speed': 'moderate'}
    logging.info(f"Using model from: {args['model_dir']}")
    logging.info(f"Saving audio to: {args['save_dir']}")
    os.makedirs(args['save_dir'], exist_ok=True)
    if platform.system() == 'Darwin' and torch.backends.mps.is_available():
        device = torch.device(f"mps:{args['device']}")
        logging.info(f'Using MPS device: {device}')
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{args['device']}")
        logging.info(f'Using CUDA device: {device}')
    else:
        device = torch.device('cpu')
        logging.info('GPU acceleration not available, using CPU')
    model = exe.create_interface_objects(interface_class_name='SparkTTS', model_dir=args['model_dir'], device=device)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(args['save_dir'], f'{timestamp}.wav')
    logging.info('Starting inference...')
    with torch.no_grad():
        wav = exe.run('inference', text=args['text'], prompt_speech_path=args['prompt_speech_path'], prompt_text=args['prompt_text'], gender=args['gender'], pitch=args['pitch'], speed=args['speed'])
        sf.write(save_path, wav, samplerate=16000)
    logging.info(f'Audio saved at: {save_path}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
run_tts()