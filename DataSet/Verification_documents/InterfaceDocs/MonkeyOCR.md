# API Documentation

## Function: `parse_folder`

### Description
The `parse_folder` function is designed to parse all PDF and image files within a specified folder. It processes these files either individually or in groups based on their total page count, utilizing a specified OCR model for recognition tasks.

### Parameters

- **`folder_path`** (`str`): 
  - **Description**: The path to the input folder containing the files to be parsed.
  - **Value Range**: Must be a valid directory path.

- **`output_dir`** (`str`): 
  - **Description**: The directory where the output files will be saved.
  - **Value Range**: Must be a valid directory path.

- **`config_path`** (`str`): 
  - **Description**: The path to the configuration file required for initializing the OCR model.
  - **Value Range**: Must be a valid file path.

- **`task`** (`str`, optional): 
  - **Description**: Specifies the type of recognition task to be performed on the files. If not provided, the function will perform a general parsing.
  - **Value Range**: Any string representing a valid task type.

- **`split_pages`** (`bool`, optional): 
  - **Description**: Indicates whether to split the pages of the files during processing.
  - **Value Range**: `True` or `False`. Default is `False`.

- **`group_size`** (`int`, optional): 
  - **Description**: The maximum number of files to group together based on their total page count. If set to `None`, files will be processed individually.
  - **Value Range**: Must be a positive integer or `None`.

- **`pred_abandon`** (`bool`, optional): 
  - **Description**: Indicates whether to abandon predictions for certain files during processing.
  - **Value Range**: `True` or `False`. Default is `False`.

### Returns
- **`str`**: The output directory path where the processed files are saved.

### Purpose
The `parse_folder` function facilitates the batch processing of PDF and image files for OCR tasks, allowing for efficient handling of multiple files while providing options for grouping and task specification. It also generates a summary of the processing results, including successful and failed file counts.

### Example Usage
```python
output_directory = parse_folder('/path/to/input/folder', '/path/to/output/folder', '/path/to/config/file', task='text_recognition', group_size=5)
```

### Notes
- The function will raise a `FileNotFoundError` if the specified folder does not exist.
- A `ValueError` will be raised if the provided `folder_path` is not a directory.
- The function prints detailed logs of the processing steps, including any errors encountered during file processing.

# API Documentation

## Function: `single_task_recognition`

### Description
The `single_task_recognition` function performs recognition on a specific content type (text, formula, or table) from a given input file. It processes the file, converts it to images if necessary, and saves the recognition results in a specified output directory.

### Parameters

- **`input_file`** (`str`): 
  - **Description**: The path to the input file (PDF or image) to be processed.
  - **Value Range**: Must be a valid file path.

- **`output_dir`** (`str`): 
  - **Description**: The directory where the output results will be saved.
  - **Value Range**: Must be a valid directory path.

- **`MonkeyOCR_model`** (`object`): 
  - **Description**: An instance of the pre-initialized OCR model used for recognition.
  - **Value Range**: Must be a valid model instance.

- **`task`** (`str`): 
  - **Description**: The type of recognition task to perform. Supported tasks include 'text', 'formula', and 'table'.
  - **Value Range**: Must be one of the following strings: `'text'`, `'formula'`, or `'table'`.

### Returns
- **`str`**: The output directory path where the recognition results are saved.

### Purpose
The `single_task_recognition` function enables targeted recognition of specific content types from input files, providing flexibility in processing and saving results in an organized manner.

### Example Usage
```python
output_directory = single_task_recognition('/path/to/input/file.pdf', '/path/to/output/folder', MonkeyOCR_model, task='text')
```

---

## Function: `parse_file`

### Description
The `parse_file` function parses a given PDF or image file and saves the results in a specified output directory. It can handle both single-page and multi-page documents, saving results accordingly.

### Parameters

- **`input_file`** (`str`): 
  - **Description**: The path to the input PDF or image file to be parsed.
  - **Value Range**: Must be a valid file path.

- **`output_dir`** (`str`): 
  - **Description**: The directory where the output results will be saved.
  - **Value Range**: Must be a valid directory path.

- **`MonkeyOCR_model`** (`object`): 
  - **Description**: An instance of the pre-initialized OCR model used for parsing.
  - **Value Range**: Must be a valid model instance.

- **`split_pages`** (`bool`, optional): 
  - **Description**: Indicates whether to split the results by pages. If `True`, each page will be processed and saved separately.
  - **Value Range**: `True` or `False`. Default is `False`.

- **`pred_abandon`** (`bool`, optional): 
  - **Description**: Indicates whether to abandon predictions for certain pages during processing.
  - **Value Range**: `True` or `False`. Default is `False`.

### Returns
- **`str`**: The output directory path where the parsing results are saved.

### Purpose
The `parse_file` function provides a comprehensive way to parse and analyze PDF and image files, allowing for detailed processing and organized output of results, including support for multi-page documents.

### Example Usage
```python
output_directory = parse_file('/path/to/input/file.pdf', '/path/to/output/folder', MonkeyOCR_model, split_pages=True)
```

