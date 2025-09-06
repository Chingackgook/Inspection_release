from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.asrt_speechrecognition import *
exe = Executor('asrt_speechrecognition', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/ASRT_SpeechRecognition/predict_speech_file.py'
import os
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage
'\n@author: nl8590687\n用于通过ASRT语音识别系统预测一次语音文件的程序\n'
import os
from speech_model import ModelSpeech
from model_zoo.speech_model.keras_backend import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    AUDIO_LENGTH = 1600
    AUDIO_FEATURE_LENGTH = 200
    CHANNELS = 1
    OUTPUT_SIZE = 1428
    sm251bn = SpeechModel251BN(input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS), output_size=OUTPUT_SIZE)
    feat = Spectrogram()
    ms = exe.create_interface_objects(interface_class_name='ModelSpeech', speech_model=sm251bn, speech_features=feat, max_label_length=64)
    ms.load_model('save_models/' + sm251bn.get_model_name() + '.model.h5')
    res = exe.run('ModelSpeech_recognize_speech_from_file', filename=RESOURCES_PATH + 'audios/test_audio.wav')
    print('*[提示] 声学模型语音识别结果：\n', res)
    ml = exe.create_interface_objects(interface_class_name='ModelLanguage', model_path='model_language')
    ml.load_model()
    str_pinyin = res
    res = exe.run('ModelLanguage_pinyin_to_text', list_pinyin=str_pinyin)
    print('语音识别最终结果：\n', res)
main()