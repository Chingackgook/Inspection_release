# API Documentation for PdfConverter

## Class: PdfConverter

The `PdfConverter` class is designed for processing and rendering PDF files into various formats such as Markdown, JSON, and HTML. It utilizes a series of processors and builders to transform the content of PDF files effectively.

### Attributes

- **override_map**: 
  - **Type**: `Dict[BlockTypes, Type[Block]]`
  - **Description**: A mapping to override the default block classes for specific block types. The keys are `BlockTypes` enum values representing the types of blocks, and the values are corresponding `Block` class implementations to use instead of the defaults.

- **use_llm**: 
  - **Type**: `bool`
  - **Description**: A flag to enable higher quality processing with LLMs (Large Language Models).

- **default_processors**: 
  - **Type**: `Tuple[BaseProcessor, ...]`
  - **Description**: A tuple of default processors that will be used for processing the document. This includes various processors for handling different content types such as text, tables, equations, etc.

### Method: `__init__`

```python
def __init__(
    self,
    artifact_dict: Dict[str, Any],
    processor_list: Optional[List[str]] = None,
    renderer: str | None = None,
    llm_service: str | None = None,
    config=None
)
```

- **Parameters**:
  - `artifact_dict` (`Dict[str, Any]`): A dictionary containing artifacts that will be used during processing.
  - `processor_list` (`Optional[List[str]]`): A list of processor class names as strings. If not provided, default processors will be used.
  - `renderer` (`str | None`): The renderer class name as a string. If not provided, the default Markdown renderer will be used.
  - `llm_service` (`str | None`): The LLM service class name as a string. If not provided, it will use the service defined in the configuration if available.
  - `config` (`Optional[Any]`): Configuration settings for the converter. If not provided, an empty dictionary will be used.

- **Return Value**: None

- **Description**: Initializes a new instance of the `PdfConverter` class, setting up the necessary processors, renderer, and LLM service based on the provided parameters and configuration.

### Method: `build_document`

```python
def build_document(self, filepath: str)
```

- **Parameters**:
  - `filepath` (`str`): The path to the PDF file that needs to be processed.

- **Return Value**: `Document`
  
- **Description**: Builds a document from the specified PDF file. It initializes the necessary builders and processes the document using the configured processors. This method is responsible for creating the document structure and applying transformations.

### Method: `__call__`

```python
def __call__(self, filepath: str)
```

- **Parameters**:
  - `filepath` (`str`): The path to the PDF file that needs to be processed.

- **Return Value**: The rendered output of the document in the specified format (e.g., Markdown, JSON, HTML).

- **Description**: This method allows the `PdfConverter` instance to be called as a function. It processes the PDF file at the given filepath and returns the rendered document. It combines the functionality of building the document and rendering it in one step.