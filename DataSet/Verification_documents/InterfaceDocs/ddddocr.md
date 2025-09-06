# API Documentation

## Class: OCREngine
The `OCREngine` class provides an interface for Optical Character Recognition (OCR) functionality. It allows users to initialize an OCR engine, process images, and retrieve recognized text.

### Attributes:
- `old` (bool): Indicates whether to use an old version of the model. Default is `False`.
- `beta` (bool): Indicates whether to use a beta version of the model. Default is `False`.
- `import_onnx_path` (str): Path to a custom ONNX model. Default is an empty string.
- `charsets_path` (str): Path to a custom character set. Default is an empty string.
- `charset_manager` (CharsetManager): Manages character sets for the OCR engine.
- `word` (bool): Indicates whether the model is word-based. Default is `False`.
- `resize` (List[int]): Specifies the image resize dimensions. Default is an empty list.
- `channel` (int): Specifies the number of channels in the image. Default is `1`.
- `is_initialized` (bool): Indicates whether the OCR engine has been successfully initialized.

### Method: `__init__`
```python
def __init__(self, use_gpu: bool = False, device_id: int = 0, 
             old: bool = False, beta: bool = False,
             import_onnx_path: str = "", charsets_path: str = "")
```
#### Parameters:
- `use_gpu` (bool): Whether to use GPU for processing. Default is `False`.
- `device_id` (int): The ID of the GPU device to use. Default is `0`.
- `old` (bool): Whether to use an old version of the model. Default is `False`.
- `beta` (bool): Whether to use a beta version of the model. Default is `False`.
- `import_onnx_path` (str): Path to a custom ONNX model. Default is an empty string.
- `charsets_path` (str): Path to a custom character set. Default is an empty string.

#### Returns:
- None

#### Description:
Initializes the OCR engine with specified parameters and prepares it for use.

### Method: `initialize`
```python
def initialize(self, **kwargs) -> None
```
#### Parameters:
- `**kwargs`: Additional keyword arguments for initialization (not specified).

#### Returns:
- None

#### Description:
Initializes the OCR engine, loading the model and character set based on the provided parameters. Raises `ModelLoadError` if initialization fails.

### Method: `predict`
```python
def predict(self, image: Union[bytes, str, Image.Image], 
            png_fix: bool = False, probability: bool = False,
            color_filter_colors: Optional[List[str]] = None,
            color_filter_custom_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
            charset_range: Optional[Union[int, str, List[str]]] = None) -> Union[str, Dict[str, Any]]
```
#### Parameters:
- `image` (Union[bytes, str, Image.Image]): The input image for OCR processing. Can be in bytes, file path, or PIL Image format.
- `png_fix` (bool): Whether to fix PNG transparency issues. Default is `False`.
- `probability` (bool): Whether to return probability information along with the recognized text. Default is `False`.
- `color_filter_colors` (Optional[List[str]]): List of preset colors for color filtering. Default is `None`.
- `color_filter_custom_ranges` (Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]]): Custom HSV color ranges for filtering. Default is `None`.
- `charset_range` (Optional[Union[int, str, List[str]]]): Character set range limit. Default is `None`.

#### Returns:
- Union[str, Dict[str, Any]]: Recognized text or a dictionary containing the text and probability information.

#### Description:
Executes OCR recognition on the provided image and returns the recognized text or additional probability information. Raises `ImageProcessError` if image processing fails or `ModelLoadError` if the engine is not initialized.

### Method: `set_charset_range`
```python
def set_charset_range(self, charset_range: Union[int, str, List[str]]) -> None
```
#### Parameters:
- `charset_range` (Union[int, str, List[str]]): The character set range to be set.

#### Returns:
- None

#### Description:
Sets the character set range for the OCR engine, allowing for customization of recognized characters.

### Method: `get_charset`
```python
def get_charset(self) -> List[str]
```
#### Returns:
- List[str]: The list of characters in the current character set.

#### Description:
Retrieves the current character set used by the OCR engine. This can be useful for understanding which characters the engine is capable of recognizing.