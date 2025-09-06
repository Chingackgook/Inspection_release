# API Documentation

## Class: Classifier

### Description
The `Classifier` class is an abstract base class for all Flair models that perform classification tasks. It provides unified functionality for both single-label and multi-label classification and evaluation, ensuring fair comparisons between multiple classifiers.

### Attributes
- **model_path**: (Union[str, Path, dict[str, Any]]) The path to the model to be loaded.
- **embedding_storage_mode**: (EmbeddingStorageMode) The mode for storing embeddings, default is 'none'.
- **mini_batch_size**: (int) The size of mini-batches for processing data, default is 32.

### Methods

#### `evaluate`
```python
def evaluate(
    self,
    data_points: Union[list[DT], Dataset],
    gold_label_type: str,
    out_path: Optional[Union[str, Path]] = None,
    embedding_storage_mode: EmbeddingStorageMode = "none",
    mini_batch_size: int = 32,
    main_evaluation_metric: tuple[str, str] = ("micro avg", "f1-score"),
    exclude_labels: Optional[list[str]] = None,
    gold_label_dictionary: Optional[Dictionary] = None,
    return_loss: bool = True,
    **kwargs,
) -> Result
```
- **Parameters**:
  - `data_points`: A list of data points or a Dataset to evaluate.
  - `gold_label_type`: The type of gold labels to evaluate against.
  - `out_path`: Optional path to save the evaluation results.
  - `embedding_storage_mode`: Mode for storing embeddings (default is 'none').
  - `mini_batch_size`: Size of mini-batches for evaluation (default is 32).
  - `main_evaluation_metric`: Tuple specifying the main evaluation metric (default is ("micro avg", "f1-score")).
  - `exclude_labels`: Optional list of labels to exclude from evaluation.
  - `gold_label_dictionary`: Optional dictionary for gold labels.
  - `return_loss`: Whether to return the loss (default is True).
  - `**kwargs`: Additional keyword arguments.

- **Returns**: 
  - A `Result` object containing evaluation metrics, detailed results, and classification report.

- **Purpose**: 
  Evaluates the model on the provided data points and computes various classification metrics.

#### `predict`
```python
@abstractmethod
def predict(
    self,
    sentences: Union[list[DT], DT],
    mini_batch_size: int = 32,
    return_probabilities_for_all_classes: bool = False,
    verbose: bool = False,
    label_name: Optional[str] = None,
    return_loss: bool = False,
    embedding_storage_mode: EmbeddingStorageMode = "none",
)
```
- **Parameters**:
  - `sentences`: The data points (most commonly Sentence objects) for which predictions are to be made.
  - `mini_batch_size`: Size of mini-batches for prediction (default is 32).
  - `return_probabilities_for_all_classes`: If True, returns probabilities for all classes (default is False).
  - `verbose`: If True, displays a progress bar (default is False).
  - `label_name`: Optional identifier for the predicted label type.
  - `return_loss`: If True, returns loss (only possible if gold labels are set).
  - `embedding_storage_mode`: Mode for storing embeddings (default is 'none').

- **Returns**: 
  - Predicted labels are added to the respective data points.

- **Purpose**: 
  Uses the model to predict labels for a given set of data points.

#### `get_used_tokens`
```python
def get_used_tokens(
    self, 
    corpus: Corpus, 
    context_length: int = 0, 
    respect_document_boundaries: bool = True
) -> typing.Iterable[list[str]]
```
- **Parameters**:
  - `corpus`: The corpus from which to extract tokens.
  - `context_length`: Length of context to consider (default is 0).
  - `respect_document_boundaries`: If True, respects document boundaries (default is True).

- **Returns**: 
  - An iterable of lists of tokens used in the corpus.

- **Purpose**: 
  Retrieves tokens from the corpus, including context tokens based on the specified length.

#### `load`
```python
@classmethod
def load(cls, model_path: Union[str, Path, dict[str, Any]]) -> "Classifier"
```
- **Parameters**:
  - `model_path`: The path to the model to be loaded.

- **Returns**: 
  - An instance of `Classifier`.

- **Purpose**: 
  Loads a classifier model from the specified path.