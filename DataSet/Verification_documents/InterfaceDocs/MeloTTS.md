# API Documentation for TTS Class

## Class: TTS

The `TTS` class implements a text-to-speech (TTS) synthesizer using a neural network model. It is designed to convert text input into audio output in various languages and tones.

### Attributes:
- **language** (str): The language code for the TTS model (e.g., 'EN', 'ZH').
- **device** (str): The device to run the model on ('cpu', 'cuda', or 'mps'). Defaults to 'auto'.
- **model** (SynthesizerTrn): The TTS model instance.
- **symbol_to_id** (dict): A mapping from symbols to their corresponding IDs.
- **hps** (object): Hyperparameters for the TTS model.
- **device** (str): The device on which the model is loaded.

### Method: __init__

```python
def __init__(self, language, device='auto', use_hf=True, config_path=None, ckpt_path=None):
```

#### Parameters:
- **language** (str): The language code for the TTS model (e.g., 'EN', 'ZH').
- **device** (str, optional): The device to run the model on. Options are 'cpu', 'cuda', or 'mps'. Defaults to 'auto'.
- **use_hf** (bool, optional): Flag to indicate whether to use Hugging Face for model loading. Defaults to True.
- **config_path** (str, optional): Path to the configuration file. Defaults to None.
- **ckpt_path** (str, optional): Path to the model checkpoint file. Defaults to None.

#### Return Value:
- None

#### Description:
Initializes the TTS model with the specified language and device. Loads the model configuration and state dictionary.

---

### Method: audio_numpy_concat

```python
@staticmethod
def audio_numpy_concat(segment_data_list, sr, speed=1.):
```

#### Parameters:
- **segment_data_list** (list of np.ndarray): A list of audio segments to concatenate.
- **sr** (int): The sampling rate of the audio.
- **speed** (float, optional): The speed factor for the audio. Defaults to 1.0. Must be greater than 0.

#### Return Value:
- np.ndarray: A single concatenated audio array.

#### Description:
Concatenates a list of audio segments into a single audio array, adding a short silence between segments. The speed factor can be used to adjust the playback speed.

---

### Method: split_sentences_into_pieces

```python
@staticmethod
def split_sentences_into_pieces(text, language, quiet=False):
```

#### Parameters:
- **text** (str): The input text to be split into sentences.
- **language** (str): The language code for the text (e.g., 'EN', 'ZH').
- **quiet** (bool, optional): If True, suppresses output messages. Defaults to False.

#### Return Value:
- list of str: A list of sentences split from the input text.

#### Description:
Splits the input text into manageable sentences based on the specified language. Optionally prints the split sentences if `quiet` is False.

---

### Method: tts_to_file

```python
def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None, format=None, position=None, quiet=False):
```

#### Parameters:
- **text** (str): The input text to be converted to speech.
- **speaker_id** (int): The ID of the speaker to use for the TTS.
- **output_path** (str, optional): The file path to save the output audio. If None, returns audio data instead. Defaults to None.
- **sdp_ratio** (float, optional): The ratio for the speech duration prediction. Defaults to 0.2. Must be between 0 and 1.
- **noise_scale** (float, optional): The scale of noise to add to the audio. Defaults to 0.6. Must be non-negative.
- **noise_scale_w** (float, optional): The weight of the noise scale. Defaults to 0.8. Must be non-negative.
- **speed** (float, optional): The speed factor for the audio. Defaults to 1.0. Must be greater than 0.
- **pbar** (callable, optional): A progress bar function to visualize progress. Defaults to None.
- **format** (str, optional): The audio format for saving (e.g., 'WAV', 'FLAC'). Defaults to None.
- **position** (int, optional): The position for the progress bar. Defaults to None.
- **quiet** (bool, optional): If True, suppresses output messages. Defaults to False.

#### Return Value:
- np.ndarray or None: Returns the generated audio as a numpy array if `output_path` is None; otherwise, saves the audio to the specified file.

#### Description:
Converts the input text into speech using the specified speaker ID and saves the output to a file or returns the audio data. Supports various parameters for audio generation customization.

