# OpenVoice 
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/OpenVoice')
# 以上是自动生成的代码，请勿修改


from typing import Any, Dict
import numpy as np
import torch
from openvoice.api import BaseSpeakerTTS , ToneColorConverter

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.speaker_tts = None
        self.tone_color_converter = None

    def create_interface_objects(self, config_path: str, model_type: str, **kwargs):
        if model_type == 'BaseSpeakerTTS':
            self.speaker_tts = BaseSpeakerTTS(config_path, **kwargs)
        elif model_type == 'ToneColorConverter':
            self.tone_color_converter = ToneColorConverter(config_path, **kwargs)

    def run(self, name: str, **kwargs):
        if name == 'BaseSpeakerTTS_tts':
            if self.speaker_tts:
                audio_output = self.speaker_tts.tts(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=kwargs.get('output_path') is not None,
                    file_path=kwargs.get('output_path', ''),
                    except_data=audio_output,
                    interface_return=audio_output
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='BaseSpeakerTTS not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
        
        elif name == 'BaseSpeakerTTS_get_text':
            if self.speaker_tts:
                text_tensor = self.speaker_tts.get_text(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=text_tensor,
                    interface_return=text_tensor
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='BaseSpeakerTTS not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'BaseSpeakerTTS_audio_numpy_concat':
            if self.speaker_tts:
                concatenated_audio = self.speaker_tts.audio_numpy_concat(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=concatenated_audio,
                    interface_return=concatenated_audio
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='BaseSpeakerTTS not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'BaseSpeakerTTS_split_sentences_into_pieces':
            if self.speaker_tts:
                sentences = self.speaker_tts.split_sentences_into_pieces(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=sentences,
                    interface_return=sentences
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='BaseSpeakerTTS not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'ToneColorConverter_extract_se':
            if self.tone_color_converter:
                features = self.tone_color_converter.extract_se(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=features,
                    interface_return=features
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='ToneColorConverter not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'ToneColorConverter_convert':
            if self.tone_color_converter:
                converted_audio = self.tone_color_converter.convert(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=kwargs.get('output_path') is not None,
                    file_path=kwargs.get('output_path', ''),
                    except_data=converted_audio,
                    interface_return=converted_audio
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='ToneColorConverter not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'ToneColorConverter_add_watermark':
            if self.tone_color_converter:
                watermarked_audio = self.tone_color_converter.add_watermark(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=watermarked_audio,
                    interface_return=watermarked_audio
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='ToneColorConverter not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )

        elif name == 'ToneColorConverter_detect_watermark':
            if self.tone_color_converter:
                detected_message = self.tone_color_converter.detect_watermark(**kwargs)
                self.result.set_result(
                    fuc_name=name,
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=detected_message,
                    interface_return=detected_message
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='ToneColorConverter not initialized',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
