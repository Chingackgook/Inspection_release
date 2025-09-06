from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Real_Time_Voice_Cloning import *
exe = Executor('Real_Time_Voice_Cloning', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/Real-Time-Voice-Cloning/demo_cli.py'
import argparse
import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import torch
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import sounddevice as sd

def run_synthesis():
    enc_model_fpath = Path('saved_models/default/encoder.pt')
    syn_model_fpath = Path('saved_models/default/synthesizer.pt')
    voc_model_fpath = Path('saved_models/default/vocoder.pt')
    in_fpath = Path(RESOURCES_PATH + 'audios/test_audio.wav')
    text = 'This is a test sentence for synthesis.'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Running a test of your configuration...\n')
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device_id)
        print('Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with %.1fGb total memory.\n' % (torch.cuda.device_count(), device_id, gpu_properties.name, gpu_properties.major, gpu_properties.minor, gpu_properties.total_memory / 1000000000.0))
    else:
        print('Using CPU for inference.\n')
    print('Preparing the encoder, the synthesizer and the vocoder...')
    ensure_default_models(Path('saved_models'))
    encoder.load_model(enc_model_fpath)
    synthesizer = exe.create_interface_objects(interface_class_name='Synthesizer', model_fpath=syn_model_fpath, verbose=True)
    vocoder.load_model(voc_model_fpath)
    print('Testing your configuration with small inputs.')
    print('\tTesting the encoder...')
    encoder.embed_utterance(np.zeros(encoder.sampling_rate))
    embed = np.random.rand(speaker_embedding_size)
    embed /= np.linalg.norm(embed)
    embeds = [embed, np.zeros(speaker_embedding_size)]
    texts = ['test 1', 'test 2']
    print('\tTesting the synthesizer... (loading the model will output a lot of text)')
    specs = exe.run('synthesize_spectrograms', texts=texts, embeddings=embeds)
    mel = np.concatenate(specs, axis=1)
    no_action = lambda *args: None
    print('\tTesting the vocoder...')
    vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
    print('All test passed! You can now synthesize speech.\n\n')
    preprocessed_wav = encoder.preprocess_wav(in_fpath)
    print('Loaded file successfully')
    embed = encoder.embed_utterance(preprocessed_wav)
    print('Created the embedding')
    texts = [text]
    embeds = [embed]
    specs = exe.run('synthesize_spectrograms', texts=texts, embeddings=embeds)
    spec = specs[0]
    print('Created the mel spectrogram')
    print('Synthesizing the waveform:')
    generated_wav = vocoder.infer_waveform(spec)
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode='constant')
    generated_wav = encoder.preprocess_wav(generated_wav)
    output_filename = os.path.join(FILE_RECORD_PATH, 'demo_output.wav')
    sf.write(output_filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    print('\nSaved output as %s\n\n' % output_filename)
run_synthesis()