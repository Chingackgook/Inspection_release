from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.bark import *
exe = Executor('bark','simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
from typing import Dict, Optional, Union
from scipy.io.wavfile import write as write_wav
from bark.api import generate_audio
from bark.generation import SAMPLE_RATE

def generate_audio_from_text():
    """Generate audio from text with predefined parameters."""
    # Parts that may need manual modification:
    input_text: str = 'Hello, this is a test audio generation.'
    output_filename: str = 'bark_generation.wav'
    output_dir: str = FILE_RECORD_PATH  # Using global variable for output path
    history_prompt: Optional[str] = None
    text_temp: float = 0.7
    waveform_temp: float = 0.7
    silent: bool = False
    output_full: bool = False
    # end

    try:
        os.makedirs(output_dir, exist_ok=True)
        generated_audio = exe.run('generate_audio', text=input_text, history_prompt=history_prompt, text_temp=text_temp, waveform_temp=waveform_temp, silent=silent, output_full=output_full)
        output_file_path = os.path.join(output_dir, output_filename)
        write_wav(output_file_path, SAMPLE_RATE, generated_audio)
        print(f"Done! Output audio file is saved at: '{output_file_path}'")
    except Exception as e:
        print(f'Oops, an error occurred: {e}')

# Run the main logic directly
generate_audio_from_text()
