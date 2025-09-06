# API Documentation

## Class: DetInferencer

The `DetInferencer` class is an interface for object detection inference, allowing users to utilize pre-trained models for detecting objects in input images. It supports visualization and saving of results.

### Attributes:
- `num_visualized_imgs` (int): A counter for the number of images processed for visualization.
- `num_predicted_imgs` (int): A counter for the number of images processed for predictions.
- `palette` (str): Color palette used for visualization.
- `show_progress` (bool): Indicates whether to display a progress bar during inference.

### Method: `__init__`

```python
def __init__(self, model: Optional[Union[ModelType, str]] = None, weights: Optional[str] = None, device: Optional[str] = None, scope: Optional[str] = 'mmdet', palette: str = 'none', show_progress: bool = True) -> None:
```

#### Parameters:
- `model` (Optional[Union[ModelType, str]]): Path to the model configuration file or model name. Default is `None`.
- `weights` (Optional[str]): Path to the checkpoint file. Default is `None`.
- `device` (Optional[str]): Device to run inference on. Default is `None`, which uses the available device.
- `scope` (Optional[str]): The scope of the model. Default is `'mmdet'`.
- `palette` (str): Color palette for visualization. Default is `'none'`.
- `show_progress` (bool): Controls the display of a progress bar during inference. Default is `True`.

#### Returns:
- None

#### Description:
Initializes the `DetInferencer` class, setting up the model, weights, device, and visualization options.

---

### Method: `preprocess`

```python
def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
```

#### Parameters:
- `inputs` (InputsType): Inputs provided by the user, which can be a string, numpy array, or a sequence of these.
- `batch_size` (int): The size of the batch for inference. Default is `1`.

#### Returns:
- `Iterable`: An iterable object that contains data processed for model input.

#### Description:
Processes the inputs into a format suitable for the model. It yields data in chunks based on the specified batch size.

---

### Method: `__call__`

```python
def __call__(self, inputs: InputsType, batch_size: int = 1, return_vis: bool = False, show: bool = False, wait_time: int = 0, no_save_vis: bool = False, draw_pred: bool = True, pred_score_thr: float = 0.3, return_datasamples: bool = False, print_result: bool = False, no_save_pred: bool = True, out_dir: str = '', **kwargs) -> dict:
```

#### Parameters:
- `inputs` (InputsType): Inputs for the inferencer.
- `batch_size` (int): Inference batch size. Default is `1`.
- `return_vis` (bool): Whether to return visualization results. Default is `False`.
- `show` (bool): Whether to display visualization results in a popup window. Default is `False`.
- `wait_time` (int): Interval for displaying images (in seconds). Default is `0`.
- `no_save_vis` (bool): Whether to avoid saving visualization results. Default is `False`.
- `draw_pred` (bool): Whether to draw predicted bounding boxes. Default is `True`.
- `pred_score_thr` (float): Minimum score for bounding boxes to be drawn. Default is `0.3`.
- `return_datasamples` (bool): Whether to return results as `DetDataSample`. Default is `False`.
- `print_result` (bool): Whether to print inference results to the console. Default is `False`.
- `no_save_pred` (bool): Whether to avoid saving prediction results. Default is `True`.
- `out_dir` (str): Directory to save inference results. Default is an empty string.
- `**kwargs`: Additional keyword arguments for preprocessing, forward, visualization, and postprocessing.

#### Returns:
- `dict`: A dictionary containing inference and visualization results.

#### Description:
Calls the inferencer to perform inference on the provided inputs, processes the results, and optionally visualizes and saves them.

---

### Method: `visualize`

```python
def visualize(self, inputs: InputsType, preds: PredType, return_vis: bool = False, show: bool = False, wait_time: int = 0, draw_pred: bool = True, pred_score_thr: float = 0.3, no_save_vis: bool = False, img_out_dir: str = '', **kwargs) -> Union[List[np.ndarray], None]:
```

#### Parameters:
- `inputs` (InputsType): Inputs for the inferencer, can be a list of image paths or numpy arrays.
- `preds` (PredType): Predictions made by the model.
- `return_vis` (bool): Whether to return visualization results. Default is `False`.
- `show` (bool): Whether to display images in a popup window. Default is `False`.
- `wait_time` (int): Interval for displaying images (in seconds). Default is `0`.
- `draw_pred` (bool): Whether to draw predicted bounding boxes. Default is `True`.
- `pred_score_thr` (float): Minimum score for bounding boxes to be drawn. Default is `0.3`.
- `no_save_vis` (bool): Whether to avoid saving visualization results. Default is `False`.
- `img_out_dir` (str): Output directory for visualization results. Default is an empty string.
- `**kwargs`: Additional keyword arguments for visualization.

#### Returns:
- `List[np.ndarray]` or `None`: Returns a list of visualization results if applicable; otherwise, returns `None`.

#### Description:
Visualizes the predictions on the input images, optionally displaying them and saving the results.

---

### Method: `postprocess`

```python
def postprocess(self, preds: PredType, visualization: Optional[List[np.ndarray]] = None, return_datasamples: bool = False, print_result: bool = False, no_save_pred: bool = False, pred_out_dir: str = '', **kwargs) -> Dict:
```

#### Parameters:
- `preds` (PredType): Predictions made by the model.
- `visualization` (Optional[List[np.ndarray]]): Visualization results.
- `return_datasamples` (bool): Whether to return results as `DetDataSample`. Default is `False`.
- `print_result` (bool): Whether to print inference results to the console. Default is `False`.
- `no_save_pred` (bool): Whether to avoid saving prediction results. Default is `False`.
- `pred_out_dir` (str): Directory to save prediction results. Default is an empty string.
- `**kwargs`: Additional keyword arguments for postprocessing.

#### Returns:
- `dict`: A dictionary containing processed predictions and visualization results.

#### Description:
Processes the predictions and visualization results, converting them into a format suitable for output, and optionally saving them.

---

### Method: `pred2dict`

```python
def pred2dict(self, data_sample: DetDataSample, pred_out_dir: str = '') -> Dict:
```

#### Parameters:
- `data_sample` (DetDataSample): Predictions of the model.
- `pred_out_dir` (str): Directory to save prediction results. Default is an empty string.

#### Returns:
- `dict`: A dictionary containing the necessary elements to represent a prediction.

#### Description:
Extracts essential elements from a prediction data sample and formats them into a dictionary, ensuring the result is JSON-serializable.