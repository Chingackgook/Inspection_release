import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.Real_Time_Voice_Cloning import ENV_DIR
from Inspection.adapters.custom_adapters.Real_Time_Voice_Cloning import *
exe = Executor('Real_Time_Voice_Cloning', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
from pathlib import Path
import librosa
import soundfile as sf
import torch
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder
import sounddevice as sd

# Hide GPUs from Pytorch to force CPU processing
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print("Running a test of your configuration...\n")

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
          "%.1fGb total memory.\n" %
          (torch.cuda.device_count(),
           device_id,
           gpu_properties.name,
           gpu_properties.major,
           gpu_properties.minor,
           gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
ensure_default_models(Path(ENV_DIR) / "saved_models")

# Load models using the exe object
exe.run('create_interface_objects', path=Path(ENV_DIR) / "saved_models/default/encoder.pt", model_type='encoder')
exe.run('create_interface_objects', path=Path(ENV_DIR) / "saved_models/default/synthesizer.pt", model_type='synthesizer')
synthesizer = exe.adapter.synthesizer
exe.run('create_interface_objects', path=Path(ENV_DIR) / "saved_models/default/vocoder.pt", model_type='vocoder')

## Run a test
print("Testing your configuration with small inputs.")
print("\tTesting the encoder...")
exe.run("embed_utterance",wav = np.zeros(encoder.sampling_rate))

# Create a dummy embedding.
embed = np.random.rand(speaker_embedding_size)
embed /= np.linalg.norm(embed)
embeds = [embed, np.zeros(speaker_embedding_size)]
texts = ["test 1", "test 2"]
print("\tTesting the synthesizer... (loading the model will output a lot of text)")
mels = exe.run("synthesize_spectrograms", texts=texts, embeddings=embeds)

# The vocoder synthesizes one waveform at a time.
mel = np.concatenate(mels, axis=1)
no_action = lambda *args: None
print("\tTesting the vocoder...")
exe.run("infer_waveform", mel=mel, target=200, overlap=50, progress_callback=no_action)

print("All test passed! You can now synthesize speech.\n\n")

## Interactive speech generation
num_generated = 0

try:
    # Get the reference audio filepath
    in_fpath = Path(ENV_DIR) / "audio/reference_audio.wav"

    ## Computing the embedding
    preprocessed_wav = exe.run("load_preprocess_wav", fpath=in_fpath)
    print("Loaded file successfully")

    embed = exe.run("embed_utterance", wav=preprocessed_wav)
    print("Created the embedding")

    ## Generating the spectrogram
    text = "我是封鹏威，来自中山大学，我在这里进行语音合成的测试。"

    texts = [text]
    embeds = [embed]
    specs = exe.run("synthesize_spectrograms", texts=texts, embeddings=embeds)
    spec = specs[0]
    print("Created the mel spectrogram")

    ## Generating the waveform
    print("Synthesizing the waveform:")
    generated_wav = exe.run("infer_waveform", mel=spec)

    # Post-generation
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    generated_wav = exe.run("postprocess_wav", fpath_or_wav=generated_wav)

    # Save it on the disk
    filename = Path(FILE_RECORD_PATH) / f"demo_output_{num_generated:02d}.wav"
    sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
    num_generated += 1
    print(f"\nSaved output as {filename}\n\n")

except Exception as e:
    print("Caught exception: %s" % repr(e))
    print("Restarting\n")
