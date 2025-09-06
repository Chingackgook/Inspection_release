# API Documentation

## Functions

### available_models

```python
def available_models() -> List[str]:
```

#### Description
Returns the names of available CLIP models.

#### Parameters
This function does not take any parameters.

#### Returns
- **List[str]**: A list of strings representing the names of available CLIP models.

---

### load

```python
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
```

#### Description
Loads a CLIP model specified by the name or path to a model checkpoint.

#### Parameters
- **name** (`str`): 
  - A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict.
  
- **device** (`Union[str, torch.device]`, optional): 
  - The device to put the loaded model. Default is `"cuda"` if a GPU is available, otherwise `"cpu"`.

- **jit** (`bool`, optional): 
  - Whether to load the optimized JIT model or the more hackable non-JIT model. Default is `False`.

- **download_root** (`str`, optional): 
  - Path to download the model files. By default, it uses `"~/.cache/clip"`.

#### Returns
- **model** (`torch.nn.Module`): 
  - The loaded CLIP model.

- **preprocess** (`Callable[[PIL.Image], torch.Tensor]`): 
  - A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input.

---

### tokenize

```python
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
```

#### Description
Returns the tokenized representation of given input string(s).

#### Parameters
- **texts** (`Union[str, List[str]]`): 
  - An input string or a list of input strings to tokenize.

- **context_length** (`int`, optional): 
  - The context length to use; all CLIP models use 77 as the context length. Default is `77`.

- **truncate** (`bool`, optional): 
  - Whether to truncate the text in case its encoding is longer than the context length. Default is `False`.

#### Returns
- **Union[torch.IntTensor, torch.LongTensor]**: 
  - A two-dimensional tensor containing the resulting tokens, shape = `[number of input strings, context_length]`. 
  - Returns `LongTensor` when the PyTorch version is <1.8.0, since older `index_select` requires indices to be long.

#### Raises
- **RuntimeError**: 
  - If the input text is too long for the specified context length and `truncate` is set to `False`.