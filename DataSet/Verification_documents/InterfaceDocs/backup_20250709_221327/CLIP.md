# API Documentation for `clip` Module

## Class: `_Tokenizer`
The `_Tokenizer` class is a simple tokenizer defined in the module, which handles text tokenization.

### Initialization
There are no specific initialization parameters as it internally manages its own tokenizer.

### Properties
- **encoder**: A dictionary that maps specific tokens to their integer representations.

### Methods
The tokenizer class typically has methods such as `encode` that convert text into tokens, though these are not explicitly detailed in the provided information.

---

## Function: `available_models`
### Description
Returns the names of available CLIP models.

### Parameters
- None

### Returns
- `List[str]`: A list of strings containing the names of available CLIP models.

### Example
```python
available_models()
```

---

## Function: `load`
### Description
Loads a CLIP model.

### Parameters
- **name** (`str`): A model name listed by `available_models()` or a path to a model checkpoint containing the `state_dict`.
- **device** (`Union[str, torch.device]`, optional): The device to put the loaded model. Default is `"cuda"` if CUDA is available, else `"cpu"`.
- **jit** (`bool`, optional): Flag to load the optimized JIT model or non-JIT model. Default is `False`.
- **download_root** (`str`, optional): Path to download the model files. Default uses `"~/.cache/clip"`.

### Returns
- `Tuple[torch.nn.Module, Callable[[PIL.Image], torch.Tensor]]`: 
  - **model**: The CLIP model as a PyTorch nn.Module.
  - **preprocess**: A callable that transforms a PIL image to a tensor compatible with the model input.

### Example
```python
model, preprocess = load("RN50")
```

---

## Function: `tokenize`
### Description
Returns the tokenized representation of given input string(s).

### Parameters
- **texts** (`Union[str, List[str]]`): An input string or a list of input strings to tokenize.
- **context_length** (`int`, optional): The context length to use; all CLIP models use 77 as default. Default value is `77`.
- **truncate** (`bool`, optional): Whether to truncate the text if its encoding exceeds the context length. Default is `False`.

### Returns
- `Union[torch.IntTensor, torch.LongTensor]`: A 2D tensor containing the resulting tokens with shape `[number of input strings, context_length]`.
  
### Example
```python
tokens = tokenize(["A human", "A cat"], context_length=77)
```

This API documentation provides a detailed insight into the `clip` module, covering its key classes and functions with clear parameter specifications, return values, and usage examples.