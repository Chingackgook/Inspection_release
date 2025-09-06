# API Documentation

## Class: SAM

### `__init__(self, model="sam_b.pt") -> None`
- **Parameters:**
  - `model` (str, optional): Path to the pre-trained SAM model file. Default is "sam_b.pt".
- **Returns:** None
- **Description:** Initializes the SAM model with a pre-trained model file.

### `predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs)`
- **Parameters:**
  - `source` (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `bboxes` (list, optional): List of bounding box coordinates for prompted segmentation. Default is None.
  - `points` (list, optional): List of points for prompted segmentation. Default is None.
  - `labels` (list, optional): List of labels for prompted segmentation. Default is None.
  - `**kwargs`: Additional keyword arguments for prediction.
- **Returns:** list
- **Description:** Performs segmentation prediction on the given image or video source.

### `__call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs)`
- **Parameters:**
  - `source` (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `bboxes` (list, optional): List of bounding box coordinates for prompted segmentation. Default is None.
  - `points` (list, optional): List of points for prompted segmentation. Default is None.
  - `labels` (list, optional): List of labels for prompted segmentation. Default is None.
  - `**kwargs`: Additional keyword arguments for prediction.
- **Returns:** list
- **Description:** Alias for the 'predict' method.

### `info(self, detailed=False, verbose=True)`
- **Parameters:**
  - `detailed` (bool, optional): If True, displays detailed information about the model. Default is False.
  - `verbose` (bool, optional): If True, displays information on the console. Default is True.
- **Returns:** tuple
- **Description:** Logs information about the SAM model.

### `task_map(self) -> dict`
- **Parameters:** None
- **Returns:** dict
- **Description:** Provides a mapping from the 'segment' task to its corresponding 'Predictor'.

### `__init__(self, model: Union[str, Path] = "yolov8n.pt", task: str = None, verbose: bool = False) -> None`
- **Parameters:**
  - `model` (Union[str, Path]): Path to the model file or model name. Default is "yolov8n.pt".
  - `task` (str, optional): The task type (e.g., "segment"). Default is None.
  - `verbose` (bool, optional): If True, enables verbose output. Default is False.
- **Returns:** None
- **Description:** Initializes the Model class, loading the specified model and setting up necessary configurations.

### `__call__(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None, stream: bool = False, **kwargs) -> list`
- **Parameters:**
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional): Input source for prediction. Default is None.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `**kwargs`: Additional keyword arguments for prediction.
- **Returns:** list
- **Description:** Calls the predict method to perform inference on the provided source.

### `is_triton_model(model: str) -> bool`
- **Parameters:**
  - `model` (str): The model URL to check.
- **Returns:** bool
- **Description:** Checks if the provided model is a Triton Server URL.

### `is_hub_model(model: str) -> bool`
- **Parameters:**
  - `model` (str): The model identifier to check.
- **Returns:** bool
- **Description:** Checks if the provided model is a HUB model.

### `reset_weights(self) -> "Model"`
- **Parameters:** None
- **Returns:** Model
- **Description:** Resets the weights of the model to their initial state.

### `load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model"`
- **Parameters:**
  - `weights` (Union[str, Path], optional): Path to the weights file. Default is "yolov8n.pt".
- **Returns:** Model
- **Description:** Loads the specified weights into the model.

### `save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None`
- **Parameters:**
  - `filename` (Union[str, Path], optional): Path to save the model. Default is "saved_model.pt".
  - `use_dill` (bool, optional): If True, uses dill for saving. Default is True.
- **Returns:** None
- **Description:** Saves the current model state to the specified file.

### `info(self, detailed: bool = False, verbose: bool = True)`
- **Parameters:**
  - `detailed` (bool, optional): If True, provides detailed information. Default is False.
  - `verbose` (bool, optional): If True, outputs information to the console. Default is True.
- **Returns:** tuple
- **Description:** Logs and returns information about the model.

### `fuse(self)`
- **Parameters:** None
- **Returns:** None
- **Description:** Fuses the model layers for optimization.

### `embed(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None, stream: bool = False, **kwargs) -> list`
- **Parameters:**
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional): Input source for embedding. Default is None.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `**kwargs`: Additional keyword arguments for embedding.
- **Returns:** list
- **Description:** Performs embedding on the provided source.

### `predict(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None, stream: bool = False, predictor=None, **kwargs) -> list`
- **Parameters:**
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional): Input source for prediction. Default is None.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `predictor` (optional): Custom predictor to use.
  - `**kwargs`: Additional keyword arguments for prediction.
- **Returns:** list
- **Description:** Performs prediction on the provided source.

### `track(self, source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None, stream: bool = False, persist: bool = False, **kwargs) -> list`
- **Parameters:**
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor], optional): Input source for tracking. Default is None.
  - `stream` (bool, optional): If True, enables real-time streaming. Default is False.
  - `persist` (bool, optional): If True, persists the tracker. Default is False.
  - `**kwargs`: Additional keyword arguments for tracking.
- **Returns:** list
- **Description:** Performs tracking on the provided source.

### `val(self, validator=None, **kwargs)`
- **Parameters:**
  - `validator` (optional): Custom validator to use.
  - `**kwargs`: Additional keyword arguments for validation.
- **Returns:** None
- **Description:** Validates the model using the specified validator.

### `benchmark(self, **kwargs)`
- **Parameters:**
  - `**kwargs`: Additional keyword arguments for benchmarking.
- **Returns:** None
- **Description:** Benchmarks the model performance.

### `export(self, **kwargs)`
- **Parameters:**
  - `**kwargs`: Additional keyword arguments for exporting the model.
- **Returns:** None
- **Description:** Exports the model to a specified format.

### `train(self, trainer=None, **kwargs)`
- **Parameters:**
  - `trainer` (optional): Custom trainer to use.
  - `**kwargs`: Additional keyword arguments for training.
- **Returns:** None
- **Description:** Trains the model using the specified trainer.

### `tune(self, use_ray=False, iterations=10, *args, **kwargs)`
- **Parameters:**
  - `use_ray` (bool, optional): If True, uses Ray for tuning. Default is False.
  - `iterations` (int, optional): Number of tuning iterations. Default is 10.
  - `*args`: Additional positional arguments.
  - `**kwargs`: Additional keyword arguments for tuning.
- **Returns:** None
- **Description:** Tunes the model hyperparameters.

### `names(self) -> list`
- **Parameters:** None
- **Returns:** list
- **Description:** Retrieves the class names associated with the loaded model.

### `device(self) -> torch.device`
- **Parameters:** None
- **Returns:** torch.device
- **Description:** Retrieves the device on which the model's parameters are allocated.

### `transforms(self)`
- **Parameters:** None
- **Returns:** object
- **Description:** Retrieves the transformations applied to the input data of the loaded model.

### `add_callback(self, event: str, func) -> None`
- **Parameters:**
  - `event` (str): The name of the event to attach the callback to.
  - `func` (callable): The callback function to be registered.
- **Returns:** None
- **Description:** Adds a callback function for a specified event.

### `clear_callback(self, event: str) -> None`
- **Parameters:**
  - `event` (str): The name of the event for which to clear the callbacks.
- **Returns:** None
- **Description:** Clears all callback functions registered for a specified event.

### `reset_callbacks(self) -> None`
- **Parameters:** None
- **Returns:** None
- **Description:** Resets all callbacks to their default functions.

### `task_map(self) -> dict`
- **Parameters:** None
- **Returns:** dict
- **Description:** Provides a mapping from the task to its corresponding model, trainer, validator, and predictor classes.