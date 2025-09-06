from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.PaddleSpeech import *
import argparse
import os
import paddle
from paddlespeech.cli.asr import ASRExecutor
from paddlespeech.cli.text import TextExecutor
exe = Executor('PaddleSpeech', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/PaddleSpeech/demos/automatic_video_subtitiles/recognize.py'

def run_asr_pipeline():
    input_audio_file = RESOURCES_PATH + 'audios/test_audio.wav'
    device = paddle.get_device()
    asr_executor = exe.create_interface_objects(interface_class_name='ASRExecutor')
    text_executor = exe.create_interface_objects(interface_class_name='TextExecutor')
    text = exe.run('ASRExecutor___call__', audio_file=os.path.abspath(os.path.expanduser(input_audio_file)), device=device)
    result = exe.run('TextExecutor___call__', text=text, task='punc', model='ernie_linear_p3_wudao', device=device)
    print('ASR Result: \n{}'.format(text))
    print('Text Result: \n{}'.format(result))
run_asr_pipeline()