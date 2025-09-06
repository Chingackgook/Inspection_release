# API Documentation

## Functions

### `chunk_text`

**Description**: Splits the input text into chunks, each with a maximum number of characters.

**Parameters**:
- `text` (str): The text to be split.
- `max_chars` (int): The maximum number of characters per chunk. Default is 135.

**Returns**: 
- List[str]: A list of text chunks.

---

### `load_vocoder`

**Description**: Loads a vocoder model for audio generation.

**Parameters**:
- `vocoder_name` (str): The name of the vocoder to load. Options include "vocos" and "bigvgan". Default is "vocos".
- `is_local` (bool): Indicates if the vocoder should be loaded from a local path. Default is False.
- `local_path` (str): The local path to the vocoder files if `is_local` is True.
- `device` (str): The device to load the model on (e.g., "cpu", "cuda"). Default is determined by the environment.
- `hf_cache_dir` (str): The directory to cache Hugging Face models.

**Returns**: 
- The loaded vocoder model.

---

### `initialize_asr_pipeline`

**Description**: Initializes the automatic speech recognition (ASR) pipeline.

**Parameters**:
- `device` (str): The device to run the ASR model on. Default is determined by the environment.
- `dtype` (torch.dtype, optional): The data type for the model. If None, it will be set based on the device.

**Returns**: 
- None

---

### `transcribe`

**Description**: Transcribes the given reference audio file into text.

**Parameters**:
- `ref_audio` (str): The path to the reference audio file.
- `language` (str, optional): The language of the audio. If None, the language will be auto-detected.

**Returns**: 
- str: The transcribed text from the audio.

---

### `load_checkpoint`

**Description**: Loads a model checkpoint for inference.

**Parameters**:
- `model`: The model object to load the checkpoint into.
- `ckpt_path` (str): The path to the checkpoint file.
- `device` (str): The device to load the model on (e.g., "cpu", "cuda").
- `dtype` (torch.dtype, optional): The data type for the model. If None, it will be set based on the device.
- `use_ema` (bool): Indicates whether to use Exponential Moving Average (EMA) weights. Default is True.

**Returns**: 
- The model with the loaded checkpoint.

---

### `load_model`

**Description**: Loads a model for inference.

**Parameters**:
- `model_cls`: The class of the model to be instantiated.
- `model_cfg` (dict): Configuration parameters for the model.
- `ckpt_path` (str): The path to the model checkpoint.
- `mel_spec_type` (str): The type of mel spectrogram to use. Default is "vocos".
- `vocab_file` (str): The path to the vocabulary file. Default is an empty string.
- `ode_method` (str): The method for ODE integration. Default is "euler".
- `use_ema` (bool): Indicates whether to use EMA weights. Default is True.
- `device` (str): The device to load the model on (e.g., "cpu", "cuda"). Default is determined by the environment.

**Returns**: 
- The loaded model object.

---

### `remove_silence_edges`

**Description**: Removes silence from the start and end of an audio segment.

**Parameters**:
- `audio`: The audio segment from which silence will be removed.
- `silence_threshold` (int): The threshold in dBFS below which audio is considered silence. Default is -42.

**Returns**: 
- The trimmed audio segment without leading and trailing silence.

---

### `preprocess_ref_audio_text`

**Description**: Preprocesses the reference audio and text for inference.

**Parameters**:
- `ref_audio_orig` (str): The path to the original reference audio file.
- `ref_text` (str): The reference text to be used.
- `show_info` (callable): A function to display information messages. Default is `print`.

**Returns**: 
- Tuple: (str, str) containing the processed reference audio path and the reference text.

---

### `infer_process`

**Description**: Main inference process that generates audio from text.

**Parameters**:
- `ref_audio` (str): The path to the reference audio file.
- `ref_text` (str): The reference text.
- `gen_text` (str): The text to be generated into audio.
- `model_obj`: The model object used for inference.
- `vocoder`: The vocoder object used for audio generation.
- `mel_spec_type` (str): The type of mel spectrogram to use. Default is "vocos".
- `show_info` (callable): A function to display information messages. Default is `print`.
- `progress`: A progress bar object for tracking progress.
- `target_rms` (float): The target root mean square (RMS) level for the generated audio. Default is 0.1.
- `cross_fade_duration` (float): The duration of cross-fading between audio segments. Default is 0.15.
- `nfe_step` (int): The number of steps for the ODE solver. Default is 32.
- `cfg_strength` (float): The strength of the classifier-free guidance. Default is 2.0.
- `sway_sampling_coef` (float): Coefficient for sway sampling. Default is -1.0.
- `speed` (float): The speed of audio generation. Default is 1.0.
- `fix_duration` (float, optional): Fixed duration for the generated audio. Default is None.
- `device` (str): The device to run the inference on (e.g., "cpu", "cuda"). Default is determined by the environment.

**Returns**: 
- The generated audio data.

---

### `infer_batch_process`

**Description**: Processes batches of text to generate audio.

**Parameters**:
- `ref_audio`: Tuple containing the reference audio tensor and sample rate.
- `ref_text` (str): The reference text.
- `gen_text_batches` (List[str]): A list of text batches to generate audio from.
- `model_obj`: The model object used for inference.
- `vocoder`: The vocoder object used for audio generation.
- `mel_spec_type` (str): The type of mel spectrogram to use. Default is "vocos".
- `progress`: A progress bar object for tracking progress.
- `target_rms` (float): The target RMS level for the generated audio. Default is 0.1.
- `cross_fade_duration` (float): The duration of cross-fading between audio segments. Default is 0.15.
- `nfe_step` (int): The number of steps for the ODE solver. Default is 32.
- `cfg_strength` (float): The strength of the classifier-free guidance. Default is 2.0.
- `sway_sampling_coef` (float): Coefficient for sway sampling. Default is -1.0.
- `speed` (float): The speed of audio generation. Default is 1.0.
- `fix_duration` (float, optional): Fixed duration for the generated audio. Default is None.
- `device` (str): The device to run the inference on (e.g., "cpu", "cuda"). Default is determined by the environment.
- `streaming` (bool): Indicates if the output should be streamed. Default is False.
- `chunk_size` (int): The size of chunks for streaming. Default is 2048.

**Returns**: 
- Yields generated audio data and sample rate.

---

### `remove_silence_for_generated_wav`

**Description**: Removes silence from the generated audio file.

**Parameters**:
- `filename` (str): The path to the generated audio file.

**Returns**: 
- None

---

### `save_spectrogram`

**Description**: Saves a spectrogram image to a specified path.

**Parameters**:
- `spectrogram` (np.ndarray): The spectrogram data to be saved.
- `path` (str): The file path where the spectrogram image will be saved.

**Returns**: 
- None

--- 

This documentation provides a comprehensive overview of the functions available in the API, including their parameters, return values, and purposes.