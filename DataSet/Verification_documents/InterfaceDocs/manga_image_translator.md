# API Documentation

## Function: `safe_get_memory_info`

### Description
Safely retrieves the current memory usage information, returning default values in case of failure.

### Parameters
- **None**

### Returns
- **Tuple[float, int]**: A tuple containing:
  - **float**: The percentage of memory used (0.0 to 100.0).
  - **int**: The amount of available memory in megabytes (MB).

### Purpose
This function is used to obtain the current memory usage statistics, which can be useful for monitoring and optimizing memory usage during operations.

---

## Function: `force_cleanup`

### Description
Performs a forced cleanup of memory by invoking garbage collection and clearing any cached memory from PyTorch.

### Parameters
- **None**

### Returns
- **None**

### Purpose
This function is intended to free up memory resources, especially in scenarios where memory usage is high, to prevent memory-related issues during processing.

---

## Class: `MangaTranslatorLocal`

### Description
A class that extends the `MangaTranslator` to provide local translation capabilities for manga images and text files.

### Attributes
- **textlines**: `List[str]` - A list to store text lines extracted from images.
- **attempts**: `Optional[int]` - The number of translation attempts to make (default is None).
- **skip_no_text**: `bool` - Flag to skip saving images with no text (default is False).
- **text_output_file**: `Optional[str]` - The file path to save text output (default is None).
- **save_quality**: `Optional[int]` - The quality of saved images (0 to 100).
- **text_regions**: `Optional[List]` - Regions of text detected in images (default is None).
- **save_text_file**: `Optional[str]` - The file path to save text translations (default is None).
- **save_text**: `Optional[bool]` - Flag to save text translations (default is None).
- **prep_manual**: `Optional[bool]` - Flag to prepare manual translations (default is None).
- **batch_size**: `int` - The number of images to process in a batch (default is 1).
- **disable_memory_optimization**: `bool` - Flag to disable memory optimization (default is False).

### Method: `__init__(self, params: dict = None)`

#### Parameters
- **params**: `Optional[dict]` - A dictionary of parameters to initialize the translator.

#### Returns
- **None**

#### Purpose
Initializes an instance of the `MangaTranslatorLocal` class with the provided parameters.

---

### Method: `translate_path(self, path: str, dest: str = None, params: dict[str, Union[int, str]] = None)`

#### Parameters
- **path**: `str` - The path to the image or folder to translate.
- **dest**: `Optional[str]` - The destination path for saving translated files (default is None).
- **params**: `Optional[dict[str, Union[int, str]]]` - Additional parameters for translation.

#### Returns
- **None**

#### Purpose
Translates an image or a folder of images recursively, saving the results to the specified destination.

---

### Method: `translate_file(self, path: str, dest: str, params: dict, config: Config)`

#### Parameters
- **path**: `str` - The path to the image or text file to translate.
- **dest**: `str` - The destination path for saving the translated file.
- **params**: `dict` - Parameters for translation.
- **config**: `Config` - Configuration object for translation settings.

#### Returns
- **bool**: Returns `True` if the translation was successful, `False` otherwise.

#### Purpose
Handles the translation of a single file, either an image or a text file, and saves the result to the specified destination.

--- 

This documentation provides a comprehensive overview of the functions and classes within the `MangaTranslatorLocal` implementation, detailing their parameters, return values, and purposes.

