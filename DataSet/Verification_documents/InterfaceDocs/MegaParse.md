# API Documentation for MegaParse

## Class: MegaParse

The `MegaParse` class provides functionality for parsing documents, particularly PDFs, using various strategies and formatters. It integrates with the Doctr and Unstructured parsers to extract text and layout information from documents.

### Attributes:
- `config` (MegaParseConfig): Configuration settings for the MegaParse instance.
- `formatters` (List[BaseFormatter] | None): A list of formatters to apply to the parsed document.
- `doctr_parser` (DoctrParser): An instance of the Doctr parser for text detection and recognition.
- `unstructured_parser` (UnstructuredParser): An instance of the Unstructured parser for handling unstructured documents.
- `layout_model` (LayoutDetector): A model for detecting the layout of the document.

### Method: `__init__`

```python
def __init__(
    self,
    formatters: List[BaseFormatter] | None = None,
    config: MegaParseConfig = MegaParseConfig(),
    unstructured_strategy: StrategyEnum = StrategyEnum.AUTO,
) -> None
```

#### Parameters:
- `formatters` (List[BaseFormatter] | None): A list of formatter instances to apply to the parsed document. Default is `None`.
- `config` (MegaParseConfig): Configuration settings for the parser. Default is a new instance of `MegaParseConfig`.
- `unstructured_strategy` (StrategyEnum): The strategy to use for unstructured parsing. Default is `StrategyEnum.AUTO`.

#### Return Value:
- None

#### Description:
Initializes a new instance of the `MegaParse` class, setting up the necessary parsers and configurations.

---

### Method: `validate_input`

```python
def validate_input(
    self,
    file_path: Path | str | None = None,
    file: IO[bytes] | None = None,
    file_extension: str | FileExtension | None = None,
) -> FileExtension
```

#### Parameters:
- `file_path` (Path | str | None): The path to the file to be parsed. Must be provided if `file` is not.
- `file` (IO[bytes] | None): A file-like object to be parsed. Must be provided if `file_path` is not.
- `file_extension` (str | FileExtension | None): The extension of the file. Required if `file` is provided.

#### Return Value:
- `FileExtension`: The validated file extension.

#### Description:
Validates the input parameters to ensure that either a file path or a file object is provided, and determines the file extension if not explicitly given.

---

### Method: `extract_page_strategies`

```python
def extract_page_strategies(
    self, pdfium_document: pdfium.PdfDocument, rast_scale: int = 2
) -> List[Page]
```

#### Parameters:
- `pdfium_document` (pdfium.PdfDocument): The PDF document to extract page strategies from.
- `rast_scale` (int): The scale factor for rasterizing pages. Default is `2`.

#### Return Value:
- `List[Page]`: A list of `Page` objects containing strategies and rasterized images.

#### Description:
Extracts page strategies from the provided PDF document by rasterizing each page and determining the text detection and layout strategy.

---

### Method: `load`

```python
def load(
    self,
    file_path: Path | str | None = None,
    file: BinaryIO | None = None,
    file_extension: str | FileExtension = "",
    strategy: StrategyEnum = StrategyEnum.AUTO,
) -> str
```

#### Parameters:
- `file_path` (Path | str | None): The path to the file to be parsed. Must be provided if `file` is not.
- `file` (BinaryIO | None): A file-like object to be parsed. Must be provided if `file_path` is not.
- `file_extension` (str | FileExtension): The extension of the file. Required if `file` is provided.
- `strategy` (StrategyEnum): The strategy to use for parsing. Default is `StrategyEnum.AUTO`.

#### Return Value:
- `str`: The result of the parsing operation, typically a string representation of the parsed document.

#### Description:
Loads and parses the specified document, applying the appropriate parsing strategy based on the file type and provided parameters.

---

### Method: `aload`

```python
async def aload(
    self,
    file_path: Path | str | None = None,
    file: BinaryIO | None = None,
    file_extension: str | FileExtension = "",
    strategy: StrategyEnum = StrategyEnum.AUTO,
) -> str | document.Document
```

#### Parameters:
- `file_path` (Path | str | None): The path to the file to be parsed. Must be provided if `file` is not.
- `file` (BinaryIO | None): A file-like object to be parsed. Must be provided if `file_path` is not.
- `file_extension` (str | FileExtension): The extension of the file. Required if `file` is provided.
- `strategy` (StrategyEnum): The strategy to use for parsing. Default is `StrategyEnum.AUTO`.

#### Return Value:
- `str | document.Document`: The result of the parsing operation, which can be a string or a `document.Document` object.

#### Description:
Asynchronously loads and parses the specified document, applying the appropriate parsing strategy based on the file type and provided parameters. This method is designed for use in asynchronous contexts.

