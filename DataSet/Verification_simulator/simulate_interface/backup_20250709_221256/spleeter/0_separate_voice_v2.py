from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.spleeter import ENV_DIR
from Inspection.adapters.custom_adapters.spleeter import *
exe = Executor('spleeter', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
adapter = 'default'
data = join(ENV_DIR, 'path/to/training_data')  # 使用 ENV_DIR 作为根路径
params_filename = join(ENV_DIR, 'path/to/params.json')  # 使用 ENV_DIR 作为根路径
verbose = True
# end

# 导入原有的包
import json
from functools import partial
from glob import glob
from itertools import product
from os.path import join
from typing import Dict, List, Optional, Tuple
from typer import Exit, Typer
from audio import Codec
from options import AudioAdapterOption
from options import AudioBitrateOption
from options import AudioCodecOption
from options import AudioDurationOption
from options import AudioInputArgument
from options import AudioInputOption
from options import AudioOffsetOption
from options import AudioOutputOption
from options import FilenameFormatOption
from options import ModelParametersOption
from options import MUSDBDirectoryOption
from options import MWFOption
from options import TrainingDataDirectoryOption
from options import VerboseOption
from options import VersionOption
from utils.logging import configure_logger
from utils.logging import logger
import tensorflow as tf
from audio.adapter import AudioAdapter
from dataset import get_training_dataset
from dataset import get_validation_dataset
from model import model_fn
from model.provider import ModelProvider
from utils.configuration import load_configuration
from audio.adapter import AudioAdapter
from separator import Separator
import numpy as np
import pandas as pd
import musdb
import museval

class SpleeterError(Exception):
    pass

class Codec:
    WAV = 'wav'

class AudioAdapter:

    @staticmethod
    def get(adapter):
        return adapter

def configure_logger(verbose):
    pass

def load_configuration(params_filename):
    return {'model_dir': 'path/to/model_dir', 'save_checkpoints_steps': 100, 'random_seed': 42, 'save_summary_steps': 10, 'train_max_steps': 1000, 'throttle_secs': 60}

class ModelProvider:

    @staticmethod
    def writeProbe(model_dir):
        pass

def get_training_dataset(params, audio_adapter, data):
    pass

def get_validation_dataset(params, audio_adapter, data):
    pass

def model_fn(features, labels, mode, params):
    pass

spleeter: Typer = Typer(add_completion=False, no_args_is_help=True, short_help='-h')

@spleeter.callback()
def default(version: bool=False) -> None:
    pass

@spleeter.command(no_args_is_help=True)
def train(adapter: str, data: str, params_filename: str, verbose: bool) -> None:
    import tensorflow as tf
    configure_logger(verbose)
    audio_adapter = AudioAdapter.get(adapter)
    params = load_configuration(params_filename)
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.45
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=params['model_dir'], params=params, config=tf.estimator.RunConfig(save_checkpoints_steps=params['save_checkpoints_steps'], tf_random_seed=params['random_seed'], save_summary_steps=params['save_summary_steps'], session_config=session_config, log_step_count_steps=10, keep_checkpoint_max=2))
    input_fn = partial(get_training_dataset, params, audio_adapter, data)
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=params['train_max_steps'])
    input_fn = partial(get_validation_dataset, params, audio_adapter, data)
    evaluation_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=None, throttle_secs=params['throttle_secs'])
    print('Start model training')
    tf.estimator.train_and_evaluate(estimator, train_spec, evaluation_spec)
    ModelProvider.writeProbe(params['model_dir'])
    print('Model training done')

@spleeter.command(no_args_is_help=True)
def separate(deprecated_files: Optional[str]=None, files: List[str]=[join(ENV_DIR, 'path/to/audio1.wav'), join(ENV_DIR, 'path/to/audio2.wav')], adapter: str='default', bitrate: str='128k', codec: Codec=Codec.WAV, duration: float=600.0, offset: float=0.0, output_path: str=FILE_RECORD_PATH, filename_format: str='{filename}/{instrument}.{codec}', params_filename: str=join(ENV_DIR, 'path/to/params.json'), mwf: bool=False, verbose: bool=True) -> None:
    from .audio.adapter import AudioAdapter
    from .separator import Separator
    configure_logger(verbose)
    audio_adapter: AudioAdapter = AudioAdapter.get(adapter)
    separator: Separator = exe.create_interface_objects(interface_class_name='Separator', params_descriptor=params_filename, MWF=mwf, multiprocess=True)
    for filename in files:
        _ = exe.run('separate_to_file', audio_descriptor=filename, destination=output_path, audio_adapter=audio_adapter, offset=offset, duration=duration, codec=codec, bitrate=bitrate, filename_format=filename_format, synchronous=False)
    _ = exe.run('join')

# 直接运行主逻辑
configure_logger(verbose)
train(adapter=adapter, data=data, params_filename=params_filename, verbose=verbose)
