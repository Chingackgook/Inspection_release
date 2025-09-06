# API Documentation for Chat Class

## Class: `Chat`
The `Chat` class provides methods for implementing a chat functionality using audio generation and text processing techniques.

### Initialization
```python
def __init__(self, logger=logging.getLogger(__name__)):
```
#### Parameters:
- `logger` (logging.Logger): A Logger object for logging purposes. Defaults to a logger instance for the current module.

### Attributes:
- `logger`: Logger instance to log messages.
- `config`: Configuration object.
- `normalizer`: Normalizer instance for text processing.
- `sha256_map`: A dictionary mapping for SHA256 values.
- `context`: An instance of GPT.Context used for managing context.

### Methods:

#### 1. `has_loaded(use_decoder=False) -> bool`
- **Parameters**:
  - `use_decoder` (bool): Determines if the decoder should be checked as loaded. Defaults to `False`.
  
- **Returns**: 
  - `bool`: Returns `True` if all required modules are loaded; otherwise, `False`.

- **Example**:
```python
chat.has_loaded()
```

#### 2. `download_models(source='local', force_redownload=False, custom_path=None) -> Optional[str]`
- **Parameters**:
  - `source` (Literal['huggingface', 'local', 'custom']): Specifies the source from which to download models. Defaults to 'local'.
  - `force_redownload` (bool): If `True`, forces redownloading of models regardless of existing files. Defaults to `False`.
  - `custom_path` (Optional[torch.serialization.FILE_LIKE]): Custom path to load models from if source is 'custom'.
  
- **Returns**:
  - `Optional[str]`: Returns the path of downloaded models or `None` if downloading fails.

- **Example**:
```python
download_path = chat.download_models(source="local")
```

#### 3. `load(source='local', force_redownload=False, compile=False, custom_path=None, device=None, coef=None, use_flash_attn=False, use_vllm=False, experimental=False) -> bool`
- **Parameters**:
  - `source` (Literal['huggingface', 'local', 'custom']): Model source. Defaults to 'local'.
  - `force_redownload` (bool): If `True`, forces model redownload. Defaults to `False`.
  - `compile` (bool): Whether to compile the models after loading. Defaults to `False`.
  - `custom_path` (Optional[torch.serialization.FILE_LIKE]): Custom path for models.
  - `device` (Optional[torch.device]): Device on which to load the models.
  - `coef` (Optional[torch.Tensor]): Coefficient tensor.
  - `use_flash_attn` (bool): Whether to use flash attention.
  - `use_vllm` (bool): Whether to use VLLM.
  - `experimental` (bool): Flag for experimental features.
  
- **Returns**:
  - `bool`: `True` if models are loaded successfully, otherwise `False`.

- **Example**:
```python
chat.load(source="local")
```

#### 4. `unload()`
- **Description**: Unloads the models and clears the initialized attributes.

- **Example**:
```python
chat.unload()
```

#### 5. `sample_random_speaker() -> str`
- **Returns**:
  - `str`: Returns a random speaker identifier.

- **Example**:
```python
random_speaker = chat.sample_random_speaker()
```

#### 6. `sample_audio_speaker(wav: Union[np.ndarray, torch.Tensor]) -> str`
- **Parameters**:
  - `wav` (Union[np.ndarray, torch.Tensor]): The audio waveform from which to encode the speaker.
  
- **Returns**:
  - `str`: Encoded representation of the audio speaker.

- **Example**:
```python
speaker_id = chat.sample_audio_speaker(audio_waveform)
```

#### 7. `infer(text, stream=False, lang=None, skip_refine_text=False, refine_text_only=False, use_decoder=True, do_text_normalization=True, do_homophone_replacement=True, split_text=True, max_split_batch=4, params_refine_text=RefineTextParams(), params_infer_code=InferCodeParams())`
- **Parameters**:
  - `text` (Union[str, List[str]]): Input text or list of texts to process.
  - `stream` (bool): If `True`, enables streaming output. Defaults to `False`.
  - `lang` (Optional[str]): Language code for text normalization.
  - `skip_refine_text` (bool): If `True`, skips the text refining step.
  - `refine_text_only` (bool): If `True`, only refines the text without further processing.
  - `use_decoder` (bool): If `True`, uses the decoder in the inference process.
  - `do_text_normalization` (bool): If `True`, normalizes text before inference.
  - `do_homophone_replacement` (bool): If `True`, replaces homophones in text.
  - `split_text` (bool): If `True`, splits the text into chunks.
  - `max_split_batch` (int): Max number of splits in text batching.
  - `params_refine_text` (RefineTextParams): Parameters for refining text.
  - `params_infer_code` (InferCodeParams): Parameters for inference coding.
  
- **Returns**:
  - `Union[List[np.ndarray], np.ndarray]`: Returns generated audio as a numpy array or list of arrays.

- **Example**:
```python
wavs = chat.infer(["Hello", "Hola"])
```

#### 8. `interrupt()`
- **Description**: Interrupts the current processing context.

- **Example**:
```python
chat.interrupt()
```

### Summary
This documentation provides an overview of the `Chat` class, including its initialization, attributes, and method interfaces. Each method is explained with parameters, return values, and example usage to clarify its functionality.