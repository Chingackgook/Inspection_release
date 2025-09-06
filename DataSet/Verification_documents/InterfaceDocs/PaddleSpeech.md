# API Documentation for ASRExecutor

## Class: ASRExecutor
The `ASRExecutor` class is designed to execute Automatic Speech Recognition (ASR) tasks. It provides methods for initializing the model, preprocessing audio input, performing inference, and postprocessing the results.

### Attributes:
- `parser`: An instance of `argparse.ArgumentParser` for command-line argument parsing.
- `max_len`: Maximum length of audio input in seconds.
- `model`: The ASR model used for inference.
- `config`: Configuration settings for the ASR model.
- `text_feature`: Text featurizer for processing text input.
- `_inputs`: Dictionary to store input tensors for the model.
- `_outputs`: Dictionary to store output results from the model.
- `change_format`: Boolean flag indicating if audio format conversion is needed.
- `sample_rate`: Sample rate of the audio input.

### Method: `__init__`
```python
def __init__(self):
```
#### Description:
Initializes the `ASRExecutor` instance, setting up the command-line argument parser and default values for various parameters.

### Method: `preprocess`
```python
def preprocess(self, model_type: str, input: Union[str, os.PathLike]):
```
#### Parameters:
- `model_type` (str): The type of ASR model being used (e.g., "deepspeech2", "conformer", "transformer").
- `input` (Union[str, os.PathLike]): Path to the audio file to be processed.

#### Return Value:
None

#### Description:
Processes the input audio file to extract features suitable for the ASR model. It reads the audio file, converts it to the required format, and prepares it for inference.

### Method: `infer`
```python
@paddle.no_grad()
def infer(self, model_type: str):
```
#### Parameters:
- `model_type` (str): The type of ASR model being used.

#### Return Value:
None

#### Description:
Performs inference using the ASR model on the preprocessed audio input. The results are stored in the `_outputs` attribute.

### Method: `postprocess`
```python
def postprocess(self) -> Union[str, os.PathLike]:
```
#### Return Value:
Union[str, os.PathLike]: The human-readable result of the ASR task, typically the transcribed text.

#### Description:
Processes the output from the inference step and returns a human-readable format, such as the transcribed text.

### Method: `download_lm`
```python
def download_lm(self, url, lm_dir, md5sum):
```
#### Parameters:
- `url` (str): The URL from which to download the language model.
- `lm_dir` (str): Directory where the language model will be saved.
- `md5sum` (str): MD5 checksum for verifying the downloaded file.

#### Return Value:
None

#### Description:
Downloads the language model from the specified URL and saves it to the designated directory, verifying the download using the provided MD5 checksum.

### Method: `execute`
```python
def execute(self, argv: List[str]) -> bool:
```
#### Parameters:
- `argv` (List[str]): List of command-line arguments.

#### Return Value:
bool: Returns `True` if execution is successful, `False` if there were exceptions.

#### Description:
Entry point for executing the ASR task from the command line. It parses the arguments, processes the input, performs inference, and handles results.

### Method: `__call__`
```python
def __call__(self, audio_file: os.PathLike, model: str='conformer_u2pp_online_wenetspeech', lang: str='zh', codeswitch: bool=False, sample_rate: int=16000, config: os.PathLike=None, ckpt_path: os.PathLike=None, decode_method: str='attention_rescoring', num_decoding_left_chunks: int=-1, force_yes: bool=False, rtf: bool=False, device=paddle.get_device()):
```
#### Parameters:
- `audio_file` (os.PathLike): Path to the audio file to be processed.
- `model` (str, optional): The ASR model to use. Default is 'conformer_u2pp_online_wenetspeech'.
- `lang` (str, optional): Language of the model. Default is 'zh'.
- `codeswitch` (bool, optional): Whether to use code-switching. Default is `False`.
- `sample_rate` (int, optional): Sample rate of the audio. Default is 16000.
- `config` (os.PathLike, optional): Path to the configuration file. Default is `None`.
- `ckpt_path` (os.PathLike, optional): Path to the model checkpoint. Default is `None`.
- `decode_method` (str, optional): Decoding method to use. Default is 'attention_rescoring'.
- `num_decoding_left_chunks` (int, optional): Number of left chunks for decoding. Default is -1.
- `force_yes` (bool, optional): If `True`, automatically accept prompts. Default is `False`.
- `rtf` (bool, optional): If `True`, show Real-time Factor (RTF). Default is `False`.
- `device` (str, optional): Device to execute the model inference. Default is the current paddle device.

#### Return Value:
Union[str, os.PathLike]: The result of the ASR task, typically the transcribed text.

#### Description:
Python API to execute the ASR task. It initializes the model, checks the audio file, preprocesses the input, performs inference, and returns the result.

# API Documentation for TextExecutor

## Class: TextExecutor
The `TextExecutor` class is designed to execute text processing tasks, specifically for punctuation restoration. It provides methods for initializing the model, preprocessing text input, performing inference, and postprocessing the results.

### Attributes:
- `parser`: An instance of `argparse.ArgumentParser` for command-line argument parsing.
- `task`: The specific text processing task (e.g., punctuation restoration).
- `model`: The text processing model used for inference.
- `tokenizer`: The tokenizer used for processing text input.
- `_inputs`: Dictionary to store input tensors for the model.
- `_outputs`: Dictionary to store output results from the model.
- `_punc_list`: List of punctuation marks used in the task.

### Method: `__init__`
```python
def __init__(self):
```
#### Description:
Initializes the `TextExecutor` instance, setting up the command-line argument parser and default values for various parameters.

### Method: `preprocess`
```python
def preprocess(self, text: Union[str, os.PathLike]):
```
#### Parameters:
- `text` (Union[str, os.PathLike]): Input text or path to a text file to be processed.

#### Return Value:
None

#### Description:
Processes the input text to prepare it for the model. It cleans the text by removing unwanted characters and tokenizes it into input IDs and segment IDs.

### Method: `infer`
```python
@paddle.no_grad()
def infer(self):
```
#### Return Value:
None

#### Description:
Performs inference using the text processing model on the preprocessed input. The results are stored in the `_outputs` attribute.

### Method: `postprocess`
```python
def postprocess(self, isNewTrainer: bool=False) -> Union[str, os.PathLike]:
```
#### Parameters:
- `isNewTrainer` (bool, optional): If `True`, indicates that a new training method is used. Default is `False`.

#### Return Value:
Union[str, os.PathLike]: The human-readable result of the text processing task, typically the restored text with punctuation.

#### Description:
Processes the output from the inference step and returns a human-readable format, such as the text with restored punctuation.

### Method: `execute`
```python
def execute(self, argv: List[str]) -> bool:
```
#### Parameters:
- `argv` (List[str]): List of command-line arguments.

#### Return Value:
bool: Returns `True` if execution is successful, `False` if there were exceptions.

#### Description:
Entry point for executing the text processing task from the command line. It parses the arguments, processes the input, performs inference, and handles results.

### Method: `__call__`
```python
def __call__(self, text: str, task: str='punc', model: str='ernie_linear_p7_wudao', lang: str='zh', config: Optional[os.PathLike]=None, ckpt_path: Optional[os.PathLike]=None, punc_vocab: Optional[os.PathLike]=None, device: str=paddle.get_device()):
```
#### Parameters:
- `text` (str): Input text to be processed.
- `task` (str, optional): The text processing task to perform. Default is 'punc'.
- `model` (str, optional): The text processing model to use. Default is 'ernie_linear_p7_wudao'.
- `lang` (str, optional): Language of the model. Default is 'zh'.
- `config` (Optional[os.PathLike], optional): Path to the configuration file. Default is `None`.
- `ckpt_path` (Optional[os.PathLike], optional): Path to the model checkpoint. Default is `None`.
- `punc_vocab` (Optional[os.PathLike], optional): Path to the vocabulary file for punctuation. Default is `None`.
- `device` (str, optional): Device to execute the model inference. Default is the current paddle device.

#### Return Value:
Union[str, os.PathLike]: The result of the text processing task, typically the restored text with punctuation.

#### Description:
Python API to execute the text processing task. It initializes the model, preprocesses the input text, performs inference, and returns the result. The method supports both old and new model versions for flexibility.

