# API Documentation

## Class: CosyVoice2

### Description
The `CosyVoice2` class is an upgraded version of `CosyVoice`, designed for advanced speech synthesis with additional features and improved performance. It provides methods for loading models, performing text-to-speech synthesis, and managing speaker information.

### Attributes
- `model_dir`: Directory where the model files are stored.
- `fp16`: Boolean indicating whether to use 16-bit floating point precision.
- `frontend`: An instance of `CosyVoiceFrontEnd` for processing input text and managing speaker information.
- `sample_rate`: The sample rate for audio output.
- `model`: An instance of `CosyVoice2Model` for performing text-to-speech synthesis.
- `instruct`: Boolean indicating if the model supports instruction-based inference.

### Method: `__init__(self, model_dir, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)`
#### Parameters
- `model_dir` (str): Path to the directory containing the model files.
- `load_jit` (bool, optional): Whether to load the JIT-compiled model. Default is `False`.
- `load_trt` (bool, optional): Whether to load the TensorRT model. Default is `False`.
- `fp16` (bool, optional): Whether to use 16-bit floating point precision. Default is `False`.
- `use_flow_cache` (bool, optional): Whether to use flow cache. Default is `False`.

#### Return Value
None

#### Purpose
Initializes the `CosyVoice2` instance by loading the model and its configurations from the specified directory.

---

### Method: `list_available_spks(self)`
#### Parameters
None

#### Return Value
- (list): A list of available speaker IDs.

#### Purpose
Returns a list of speaker IDs that are available for synthesis.

---

### Method: `add_zero_shot_spk(self, prompt_text, prompt_speech_16k, zero_shot_spk_id)`
#### Parameters
- `prompt_text` (str): The text prompt used for zero-shot speaker creation.
- `prompt_speech_16k` (Tensor): The audio tensor of the prompt speech at 16kHz.
- `zero_shot_spk_id` (str): The ID to assign to the new zero-shot speaker.

#### Return Value
- (bool): Returns `True` if the speaker was successfully added.

#### Purpose
Adds a new zero-shot speaker based on the provided text and speech prompt.

---

### Method: `save_spkinfo(self)`
#### Parameters
None

#### Return Value
None

#### Purpose
Saves the speaker information to a file for future use.

---

### Method: `inference_sft(self, tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)`
#### Parameters
- `tts_text` (str): The text to be synthesized.
- `spk_id` (str): The ID of the speaker to use for synthesis.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.
- `text_frontend` (bool, optional): Whether to use the text frontend for processing. Default is `True`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs speech synthesis for the given text using the specified speaker.

---

### Method: `inference_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
#### Parameters
- `tts_text` (str): The text to be synthesized.
- `prompt_text` (str): The text prompt for zero-shot synthesis.
- `prompt_speech_16k` (Tensor): The audio tensor of the prompt speech at 16kHz.
- `zero_shot_spk_id` (str, optional): The ID of the zero-shot speaker. Default is an empty string.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.
- `text_frontend` (bool, optional): Whether to use the text frontend for processing. Default is `True`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs zero-shot speech synthesis using the provided text and prompt.

---

### Method: `inference_cross_lingual(self, tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
#### Parameters
- `tts_text` (str): The text to be synthesized.
- `prompt_speech_16k` (Tensor): The audio tensor of the prompt speech at 16kHz.
- `zero_shot_spk_id` (str, optional): The ID of the zero-shot speaker. Default is an empty string.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.
- `text_frontend` (bool, optional): Whether to use the text frontend for processing. Default is `True`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs cross-lingual speech synthesis using the provided text and prompt.

---

### Method: `inference_instruct(self, tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True)`
#### Parameters
- `tts_text` (str): The text to be synthesized.
- `spk_id` (str): The ID of the speaker to use for synthesis.
- `instruct_text` (str): The instruction text for guiding synthesis.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.
- `text_frontend` (bool, optional): Whether to use the text frontend for processing. Default is `True`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs instruction-based speech synthesis using the provided text and speaker.

---

### Method: `inference_vc(self, source_speech_16k, prompt_speech_16k, stream=False, speed=1.0)`
#### Parameters
- `source_speech_16k` (Tensor): The audio tensor of the source speech at 16kHz.
- `prompt_speech_16k` (Tensor): The audio tensor of the prompt speech at 16kHz.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs voice conversion using the provided source and prompt speech.

---

### Method: `inference_instruct2(self, tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
#### Parameters
- `tts_text` (str): The text to be synthesized.
- `instruct_text` (str): The instruction text for guiding synthesis.
- `prompt_speech_16k` (Tensor): The audio tensor of the prompt speech at 16kHz.
- `zero_shot_spk_id` (str, optional): The ID of the zero-shot speaker. Default is an empty string.
- `stream` (bool, optional): Whether to stream the output. Default is `False`.
- `speed` (float, optional): The speed of the speech synthesis. Default is `1.0`.
- `text_frontend` (bool, optional): Whether to use the text frontend for processing. Default is `True`.

#### Return Value
- (Generator): Yields model output containing synthesized speech.

#### Purpose
Performs instruction-based speech synthesis using the provided text, instruction, and prompt. This method is specific to `CosyVoice2`.