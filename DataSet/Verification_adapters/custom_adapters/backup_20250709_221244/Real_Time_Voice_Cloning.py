# Real_Time_Voice_Cloning 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'Real_Time_Voice_Cloning/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/Real-Time-Voice-Cloning/')

# 可以在此位置后添加导包部分代码
from encoder import inference as encoder
from encoder.params_model import model_embedding_size as speaker_embedding_size
from synthesizer.inference import Synthesizer
from utils.argutils import print_args
from utils.default_models import ensure_default_models
from vocoder import inference as vocoder

# DeadCodeFront end

from pathlib import Path

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.synthesizer = None
        self.vocoder = None

    def create_interface_objects(self, **kwargs):
        path = kwargs.get('path', None)
        model_type = kwargs.get('model_type', None)
        if model_type == 'encoder':
            self.encoder = encoder
            self.encoder.create_interface_objects(path)
        elif model_type == 'synthesizer':
            self.synthesizer = Synthesizer(Path(path))
        elif model_type == 'vocoder':
            self.vocoder = vocoder
            self.vocoder.create_interface_objects(path)
        else:
            self.result.set_result('create_interface_objects', False, 'Invalid model type', False, '', None, None)
        self.result.set_result('create_interface_objects', True, '', False, '', None, None)


        # if 'encoder' in kwargs:
        #     self.encoder = load_model(kwargs['encoder'])
        # if 'synthesizer' in kwargs:
        #     self.synthesizer = Synthesizer(Path(kwargs['synthesizer']))
        # if 'vocoder' in kwargs:
        #     self.vocoder = load_model(kwargs['vocoder'])

    def run(self, name: str, **kwargs):
        try:
            if name == 'create_interface_objects':
                self.create_interface_objects(**kwargs)
            elif name == 'encoder_is_loaded':
                if self.encoder.is_loaded():
                    self.result.set_result(name, True, '', False, '', None, True)
                else:
                    self.result.set_result(name, False, 'Encoder model not loaded', False, '', None, False)
            elif name == 'embed_frames_batch':
                embeddings = self.encoder.embed_frames_batch(**kwargs)
                self.result.set_result(name, True, '', False, '', embeddings, embeddings)
            elif name == 'compute_partial_slices':
                slices = self.encoder.compute_partial_slices(**kwargs)
                self.result.set_result(name, True, '', False, '', slices, slices)
            elif name == 'embed_utterance':
                embedding = self.encoder.embed_utterance(**kwargs)
                self.result.set_result(name, True, '', False, '', embedding, embedding)
            elif name == 'plot_embedding_as_heatmap':
                self.encoder.plot_embedding_as_heatmap(**kwargs)
                self.result.set_result(name, True, '', False, '', None, None)
            elif name == 'process_wav':
                fpath_or_wav = kwargs.get('fpath_or_wav')
                source_sr = kwargs.get('source_sr', None)
                normalize = kwargs.get('normalize', True)
                trim_silence = kwargs.get('trim_silence', True)
                wav = self.encoder.preprocess_wav(
                    fpath_or_wav,
                    source_sr=source_sr,
                    normalize=normalize,
                    trim_silence=trim_silence
                )
                self.result.set_result(name, True, '', False, '', wav, wav)

            elif name == 'synthesizer_is_loaded':
                if self.synthesizer.is_loaded():
                    self.result.set_result(name, True, '', False, '', None, True)
                else:
                    self.result.set_result(name, False, 'Synthesizer model not loaded', False, '', None, False)
            elif name == 'synthesize_spectrograms':
                spectrograms = self.synthesizer.synthesize_spectrograms(**kwargs)
                self.result.set_result(name, True, '', False, '', spectrograms, spectrograms)
            elif name == 'load_preprocess_wav':
                wav = self.synthesizer.load_preprocess_wav(**kwargs)
                self.result.set_result(name, True, '', False, '', wav, wav)
            elif name == 'make_spectrogram':
                mel_spectrogram = self.synthesizer.make_spectrogram(**kwargs)
                self.result.set_result(name, True, '', False, '', mel_spectrogram, mel_spectrogram)
            elif name == 'griffin_lim':
                waveform = self.synthesizer.griffin_lim(**kwargs)
                self.result.set_result(name, True, '', False, '', waveform, waveform)

            elif name == 'vocoder_load_model':
                self.vocoder.create_interface_objects(**kwargs)
                self.result.set_result(name, True, '', False, '', None, None)
            elif name == 'vocoder_is_loaded':
                if self.vocoder.is_loaded():
                    self.result.set_result(name, True, '', False, '', None, True)
                else:
                    self.result.set_result(name, False, 'Vocoder model not loaded', False, '', None, False)
            elif name == 'infer_waveform':
                waveform = self.vocoder.infer_waveform(**kwargs)
                self.result.set_result(name, True, '', False, '', waveform, waveform)
        except Exception as e:
            self.result.set_result(name, False, str(e), False, '', None, None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('create_interface_objects')
adapter_additional_data['functions'].append('encoder_is_loaded')
adapter_additional_data['functions'].append('embed_frames_batch')
adapter_additional_data['functions'].append('compute_partial_slices')
adapter_additional_data['functions'].append('embed_utterance')
adapter_additional_data['functions'].append('plot_embedding_as_heatmap')
adapter_additional_data['functions'].append('synthesizer_is_loaded')
adapter_additional_data['functions'].append('synthesize_spectrograms')
adapter_additional_data['functions'].append('load_preprocess_wav')
adapter_additional_data['functions'].append('make_spectrogram')
adapter_additional_data['functions'].append('griffin_lim')
adapter_additional_data['functions'].append('vocoder_load_model')
adapter_additional_data['functions'].append('vocoder_is_loaded')
adapter_additional_data['functions'].append('infer_waveform')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
