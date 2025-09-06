# API Documentation for Magika

## Class: Magika

The `Magika` class provides an interface for identifying content types of files and streams using a deep learning model. It supports various input methods, including file paths, byte streams, and binary streams.

### `__init__`

```python
def __init__(
    self,
    model_dir: Optional[Path] = None,
    prediction_mode: PredictionMode = PredictionMode.HIGH_CONFIDENCE,
    no_dereference: bool = False,
    verbose: bool = False,
    debug: bool = False,
    use_colors: bool = False,
) -> None
```

#### Parameters:
- `model_dir` (Optional[Path]): The directory where the model files are located. If not provided, a default model directory will be used.
- `prediction_mode` (PredictionMode): The mode of prediction, which can be `HIGH_CONFIDENCE`, `MEDIUM_CONFIDENCE`, or `BEST_GUESS`. Default is `HIGH_CONFIDENCE`.
- `no_dereference` (bool): If set to `True`, symbolic links will not be dereferenced. Default is `False`.
- `verbose` (bool): If set to `True`, enables verbose logging. Default is `False`.
- `debug` (bool): If set to `True`, enables debug logging. Default is `False`.
- `use_colors` (bool): If set to `True`, enables colored logging output. Default is `False`.

#### Return Value:
- None

#### Purpose:
Initializes the `Magika` instance, loading the model and configuration files from the specified directory or the default directory.

---

### `get_module_version`

```python
def get_module_version() -> str
```

#### Parameters:
- None

#### Return Value:
- `str`: The version of the module.

#### Purpose:
Returns the version of the `Magika` module.

---

### `get_model_name`

```python
def get_model_name() -> str
```

#### Parameters:
- None

#### Return Value:
- `str`: The name of the model being used.

#### Purpose:
Returns the name of the model directory.

---

### `identify_path`

```python
def identify_path(path: Union[str, os.PathLike]) -> MagikaResult
```

#### Parameters:
- `path` (Union[str, os.PathLike]): The path to the file whose content type is to be identified.

#### Return Value:
- `MagikaResult`: An object containing the result of the content type identification.

#### Purpose:
Identifies the content type of a file given its path.

---

### `identify_paths`

```python
def identify_paths(paths: Sequence[Union[str, os.PathLike]]) -> List[MagikaResult]
```

#### Parameters:
- `paths` (Sequence[Union[str, os.PathLike]]): A sequence of paths to the files whose content types are to be identified.

#### Return Value:
- `List[MagikaResult]`: A list of results for each file path provided.

#### Purpose:
Identifies the content types of a list of files given their paths.

---

### `identify_bytes`

```python
def identify_bytes(content: bytes) -> MagikaResult
```

#### Parameters:
- `content` (bytes): The raw bytes of the content whose type is to be identified.

#### Return Value:
- `MagikaResult`: An object containing the result of the content type identification.

#### Purpose:
Identifies the content type of raw bytes.

---

### `identify_stream`

```python
def identify_stream(stream: BinaryIO) -> MagikaResult
```

#### Parameters:
- `stream` (BinaryIO): A readable binary stream whose content type is to be identified.

#### Return Value:
- `MagikaResult`: An object containing the result of the content type identification.

#### Purpose:
Identifies the content type of a `BinaryIO` stream. The method will seek around the stream as needed.

---

### `get_output_content_types`

```python
def get_output_content_types() -> List[ContentTypeLabel]
```

#### Parameters:
- None

#### Return Value:
- `List[ContentTypeLabel]`: A list of all possible output content types of the module.

#### Purpose:
Returns the list of all possible output content types of the module, considering the model's outputs and additional configurations.

---

### `get_model_content_types`

```python
def get_model_content_types() -> List[ContentTypeLabel]
```

#### Parameters:
- None

#### Return Value:
- `List[ContentTypeLabel]`: A list of all possible outputs of the underlying model.

#### Purpose:
Returns the list of all possible outputs of the underlying model, useful for debugging purposes.

