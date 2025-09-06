import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.spleeter import ENV_DIR
from Inspection.adapters.custom_adapters.spleeter import *
exe = Executor('spleeter', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包

"""
Python oneliner script usage.

USAGE: python -m spleeter {train,evaluate,separate} ...

Notes:
    All critical import involving TF, numpy or Pandas are deported to
    command function scope to avoid heavy import on CLI evaluation,
    leading to large bootstraping time.
"""
import json
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from typing import Dict, List, Optional, Tuple

# pyright: reportMissingImports=false
# pylint: disable=import-error
from typer import Exit, Typer

from spleeter import SpleeterError
from spleeter.audio import Codec
from spleeter.options import (
    AudioAdapterOption,
    AudioBitrateOption,
    AudioCodecOption,
    AudioDurationOption,
    AudioInputArgument,
    AudioInputOption,
    AudioOffsetOption,
    AudioOutputOption,
    FilenameFormatOption,
    ModelParametersOption,
    MUSDBDirectoryOption,
    MWFOption,
    TrainingDataDirectoryOption,
    VerboseOption,
    VersionOption,
)
from spleeter.utils.logging import configure_logger, logger

# pylint: enable=import-error

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help="-h")
""" CLI application. """


def run_spleeter(mock_params):
    configure_logger(mock_params['verbose'])
    params_descriptor = "spleeter:2stems"
    # Load the model using exe
    exe.create_interface_objects(params_descriptor=params_descriptor,multiprocess=False)

    # Simulate the adapter and separator initialization
    audio_adapter = AudioAdapter.default()
    

    # Simulate the separate function
    for filename in mock_params['files']:
        exe.run('separate_to_file',audio_descriptor=filename,
            destination=join(FILE_RECORD_PATH, mock_params['output_path']),
            audio_adapter=audio_adapter,
            offset=mock_params['offset'],
            duration=mock_params['duration'],
            codec=mock_params['codec'],
            bitrate=mock_params['bitrate'],
            filename_format=mock_params['filename_format'],
            synchronous=False)
        
    
    exe.run('join')

    # Simulate the evaluation if needed
    # This part can be added if you want to evaluate as well


# Mock parameters to simulate user input
mock_params = {
    "adapter": "default_adapter",
    "data": join(ENV_DIR, "path/to/training/data"),
    "params_filename": "path/to/params.json",
    "verbose": True,
    "files": [join(ENV_DIR, "file1.mp3"), join(ENV_DIR, "file2.mp3")],
    "output_path": "path/to/output",
    "bitrate": "128k",
    "codec": Codec.WAV,
    "duration": 600.0,
    "offset": 0.0,
    "filename_format": "{foldername}/{instrument}.{codec}",
    "mwf": False,
}

# Call the run_spleeter function with mock parameters
run_spleeter(mock_params)

# The following functions remain unchanged
def create_estimator(params, MWF):
    return exe.run("create_estimator", params=params, MWF=MWF)

def _get_prediction_generator(data):
    return exe.run("_get_prediction_generator", data=data)

def _get_input_provider():
    return exe.run("_get_input_provider")

def _get_features():
    return exe.run("_get_features")

def _get_builder():
    return exe.run("_get_builder")

def _get_session():
    return exe.run("_get_session")

def _separate_tensorflow(waveform, audio_descriptor):
    return exe.run("_separate_tensorflow", waveform=waveform, audio_descriptor=audio_descriptor)

# Other functions remain unchanged
