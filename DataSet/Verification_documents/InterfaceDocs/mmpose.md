# API Documentation for MMPoseInferencer

## Class: MMPoseInferencer

### Description
`MMPoseInferencer` is a unified inferencer interface for pose estimation tasks, currently supporting 2D keypoint detection. It allows users to perform inference using various pose estimation models.

### Attributes
- **visualizer**: (Optional) A visualizer instance for visualizing predictions.
- **show_progress**: (bool) Flag to indicate whether to show progress during inference.
- **inferencer**: An instance of the appropriate inferencer class (e.g., `Pose2DInferencer`, `Pose3DInferencer`, or `Hand3DInferencer`) based on the provided model type.

### Methods

#### `__init__`
```python
def __init__(self,
             pose2d: Optional[str] = None,
             pose2d_weights: Optional[str] = None,
             pose3d: Optional[str] = None,
             pose3d_weights: Optional[str] = None,
             device: Optional[str] = None,
             scope: str = 'mmpose',
             det_model: Optional[Union[ModelType, str]] = None,
             det_weights: Optional[str] = None,
             det_cat_ids: Optional[Union[int, List]] = None,
             show_progress: bool = False) -> None:
```
- **Parameters**:
  - `pose2d` (Optional[str]): Pretrained 2D pose estimation algorithm (model alias, config name, or config path). Default is `None`.
  - `pose2d_weights` (Optional[str]): Path to the custom checkpoint file for the selected pose2d model. Default is `None`.
  - `pose3d` (Optional[str]): Pretrained 3D pose estimation algorithm. Default is `None`.
  - `pose3d_weights` (Optional[str]): Path to the custom checkpoint file for the selected pose3d model. Default is `None`.
  - `device` (Optional[str]): Device to run inference (e.g., 'cpu' or 'cuda'). Default is `None`.
  - `scope` (str): The scope of the model. Default is `'mmpose'`.
  - `det_model` (Optional[Union[ModelType, str]]): Config path or alias of the detection model. Default is `None`.
  - `det_weights` (Optional[str]): Path to the checkpoints of the detection model. Default is `None`.
  - `det_cat_ids` (Optional[Union[int, List]]): Category id for the detection model. Default is `None`.
  - `show_progress` (bool): Flag to show progress during inference. Default is `False`.

- **Returns**: None
- **Purpose**: Initializes the `MMPoseInferencer` instance and sets up the appropriate inferencer based on the provided model configurations.

---

#### `preprocess`
```python
def preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs):
```
- **Parameters**:
  - `inputs` (InputsType): Inputs provided by the user (can be a string or numpy array).
  - `batch_size` (int): Batch size for processing. Default is `1`.
  - `**kwargs`: Additional keyword arguments for preprocessing.

- **Returns**: 
  - Yields processed data that is ready for model inference.
- **Purpose**: Processes the inputs into a format that can be fed into the model for inference.

---

#### `forward`
```python
@torch.no_grad()
def forward(self, inputs: InputType, **forward_kwargs) -> PredType:
```
- **Parameters**:
  - `inputs` (InputType): The inputs to be forwarded to the model (can be a string or numpy array).
  - `**forward_kwargs`: Additional keyword arguments for the forward pass.

- **Returns**: 
  - `Dict`: The prediction results, which may include keys such as "pose2d".
- **Purpose**: Forwards the processed inputs to the model and retrieves the prediction results.

---

#### `__call__`
```python
def __call__(self,
             inputs: InputsType,
             return_datasamples: bool = False,
             batch_size: int = 1,
             out_dir: Optional[str] = None,
             **kwargs) -> dict:
```
- **Parameters**:
  - `inputs` (InputsType): Inputs for the inferencer (can be a string or numpy array).
  - `return_datasamples` (bool): Whether to return results as `BaseDataElement`. Default is `False`.
  - `batch_size` (int): Batch size for processing. Default is `1`.
  - `out_dir` (Optional[str]): Directory to save visualization results and predictions. Default is `None`.
  - `**kwargs`: Additional keyword arguments for preprocessing, forward, visualization, and postprocessing.

- **Returns**: 
  - `dict`: Inference and visualization results.
- **Purpose**: Calls the inferencer to process the inputs, perform inference, visualize results, and return the final output.

---

#### `visualize`
```python
def visualize(self, inputs: InputsType, preds: PredType, **kwargs) -> List[np.ndarray]:
```
- **Parameters**:
  - `inputs` (InputsType): Inputs preprocessed for visualization (can be a list of strings or numpy arrays).
  - `preds` (PredType): Predictions from the model.
  - `**kwargs`: Additional keyword arguments for visualization settings.

- **Returns**: 
  - `List[np.ndarray]`: Visualization results as a list of numpy arrays.
- **Purpose**: Visualizes the predictions on the input images and returns the visualized results.

--- 

This documentation provides a comprehensive overview of the `MMPoseInferencer` class and its methods, detailing their parameters, return values, and purposes.

