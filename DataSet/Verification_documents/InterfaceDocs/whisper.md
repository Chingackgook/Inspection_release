# API Documentation

## Functions

### available_models

```python
def available_models() -> List[str]:
```

#### Description
Returns the names of available Whisper ASR models.

#### Parameters
- **None**

#### Returns
- **List[str]**: A list of strings representing the names of available models.

---

### load_model

```python
def load_model(
    name: str,
    device: Optional[Union[str, torch.device]] = None,
    download_root: str = None,
    in_memory: bool = False,
) -> Whisper:
```

#### Description
Loads a Whisper ASR model.

#### Parameters
- **name** (`str`): 
  - The name of the model to load. It should be one of the official model names listed by `available_models()`, or a path to a model checkpoint containing the model dimensions and the model state_dict.
  
- **device** (`Optional[Union[str, torch.device]]`): 
  - The PyTorch device to put the model into. If not specified, it defaults to "cuda" if available, otherwise "cpu".

- **download_root** (`str`): 
  - The path to download the model files. By default, it uses "~/.cache/whisper".

- **in_memory** (`bool`): 
  - Whether to preload the model weights into host memory. Defaults to `False`.

#### Returns
- **Whisper**: An instance of the Whisper ASR model.

#### Example Usage
```python
model = load_model("small")
```

---

### transcribe

```python
def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    carry_initial_prompt: bool = False,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    **decode_options,
) -> dict:
```

#### Description
Transcribes an audio file using the Whisper ASR model.

#### Parameters
- **model** (`Whisper`): 
  - The Whisper model instance to use for transcription.

- **audio** (`Union[str, np.ndarray, torch.Tensor]`): 
  - The path to the audio file to open, or the audio waveform.

- **verbose** (`Optional[bool]`): 
  - Whether to display the text being decoded to the console. If `True`, displays all details; if `False`, displays minimal details; if `None`, does not display anything.

- **temperature** (`Union[float, Tuple[float, ...]]`): 
  - Temperature for sampling. It can be a tuple of temperatures, which will be used successively upon failures.

- **compression_ratio_threshold** (`Optional[float]`): 
  - If the gzip compression ratio is above this value, treat as failed. Default is `2.4`.

- **logprob_threshold** (`Optional[float]`): 
  - If the average log probability over sampled tokens is below this value, treat as failed. Default is `-1.0`.

- **no_speech_threshold** (`Optional[float]`): 
  - If the no-speech probability is higher than this value AND the average log probability over sampled tokens is below `logprob_threshold`, consider the segment as silent. Default is `0.6`.

- **condition_on_previous_text** (`bool`): 
  - If `True`, the previous output of the model is provided as a prompt for the next window. Default is `True`.

- **initial_prompt** (`Optional[str]`): 
  - Optional text to provide as a prompt for the first window.

- **carry_initial_prompt** (`bool`): 
  - If `True`, `initial_prompt` is prepended to the prompt of each internal `decode()` call. Default is `False`.

- **word_timestamps** (`bool`): 
  - Extract word-level timestamps using the cross-attention pattern. Default is `False`.

- **prepend_punctuations** (`str`): 
  - If `word_timestamps` is `True`, merge these punctuation symbols with the next word. Default is `"\"'“¿([{-"`.

- **append_punctuations** (`str`): 
  - If `word_timestamps` is `True`, merge these punctuation symbols with the previous word. Default is `"\"'.。,，!！?？:：”)]}、"`.

- **clip_timestamps** (`Union[str, List[float]]`): 
  - Comma-separated list of start,end timestamps (in seconds) of clips to process. Default is `"0"`.

- **hallucination_silence_threshold** (`Optional[float]`): 
  - When `word_timestamps` is `True`, skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected.

- **decode_options** (`dict`): 
  - Additional keyword arguments to construct `DecodingOptions` instances.

#### Returns
- **dict**: A dictionary containing:
  - `"text"`: The resulting transcribed text.
  - `"segments"`: Segment-level details.
  - `"language"`: The detected spoken language.

#### Example Usage
```python
result = transcribe(model, "audio_file.wav")
print(result["text"])
```

---

### cli

```python
def cli():
```

#### Description
Command-line interface for transcribing audio files using the Whisper ASR model.

#### Parameters
- **None**

#### Returns
- **None**

#### Example Usage
This function is intended to be run from the command line and does not return a value. It processes audio files specified as command-line arguments.

#### Command-Line Example
```bash
python script.py audio_file.wav --model small --output_dir ./transcriptions
```

---

This documentation provides a comprehensive overview of the available functions in the Whisper ASR model implementation, detailing their parameters, return values, and usage examples.