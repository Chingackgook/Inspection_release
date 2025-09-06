# fish_speech_fixed 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'fish_speech_fixed/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/fish-speech')
# 以上是自动生成的代码，请勿修改

from fish_speech.models.vqgan.inference import load_model


class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.model = None
        self.cfg = None
        self.global_hydra = None

    def create_interface_objects(self, config_name: str, checkpoint_path: str, device: str = "cuda"):
        try:
            # Load the model using the provided function
            self.model = load_model(config_name=config_name, checkpoint_path=checkpoint_path, device=device)
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.model
            )
        except Exception as e:
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

    def run(self, name: str, **kwargs):
        try:
            if name == 'encode':
                audios = kwargs.get('audios')
                audio_lengths = kwargs.get('audio_lengths')
                features = self.model.encode(audios, audio_lengths)
                self.result.set_result(
                    fuc_name='encode',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=features,
                    interface_return=features
                )
            elif name == 'decode':
                indices = kwargs.get('indices')
                feature_lengths = kwargs.get('feature_lengths')
                decoded_audios = self.model.decode(indices, feature_lengths)
                self.result.set_result(
                    fuc_name='decode',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=decoded_audios,
                    interface_return=decoded_audios
                )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='Function not found',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
        except Exception as e:
            self.result.set_result(
                fuc_name=name,
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('encode')
adapter_additional_data['functions'].append('decode')
adapter_additional_data['functions'].append('GlobalHydra_clear')
adapter_additional_data['functions'].append('cfg_instantiate')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
