# API Documentation

## Class: Detections

### Initializer
```python
def __init__(self, xyxy: np.ndarray, mask: Optional[np.ndarray] = None, confidence: Optional[np.ndarray] = None, class_id: Optional[np.ndarray] = None, tracker_id: Optional[np.ndarray] = None, data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict), metadata: Dict[str, Any] = field(default_factory=dict)
```
- **Parameters:**
  - `xyxy` (np.ndarray): Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
  - `mask` (Optional[np.ndarray]): Optional mask for each detection.
  - `confidence` (Optional[np.ndarray]): Confidence scores for each detection.
  - `class_id` (Optional[np.ndarray]): Class IDs for each detection.
  - `tracker_id` (Optional[np.ndarray]): Tracker IDs for each detection.
  - `data` (Dict[str, Union[np.ndarray, List]]): Additional data associated with detections.
  - `metadata` (Dict[str, Any]): Metadata associated with detections.

- **Returns:** None
- **Purpose:** Initializes a Detections object with the provided detection data.

### Attributes
- `xyxy`: Bounding box coordinates.
- `mask`: Detection masks (if provided).
- `confidence`: Confidence scores for detections.
- `class_id`: Class IDs for detections.
- `tracker_id`: Tracker IDs for detections.
- `data`: Additional data associated with detections.
- `metadata`: Metadata associated with detections.

### Methods

#### `__len__`
```python
def __len__(self) -> int
```
- **Returns:** int - The number of detections in the Detections object.
- **Purpose:** Returns the count of detections.

#### `__iter__`
```python
def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[int], Optional[int], Dict[str, Union[np.ndarray, List]]]]
```
- **Returns:** Iterator - Yields tuples of detection attributes for each detection.
- **Purpose:** Iterates over the Detections object.

#### `__eq__`
```python
def __eq__(self, other: Detections) -> bool
```
- **Parameters:**
  - `other` (Detections): Another Detections object to compare.
  
- **Returns:** bool - True if both Detections objects are equal, False otherwise.
- **Purpose:** Compares two Detections objects for equality.

#### `from_yolov5`
```python
@classmethod
def from_yolov5(cls, yolov5_results) -> Detections
```
- **Parameters:**
  - `yolov5_results`: The output Detections instance from YOLOv5.
  
- **Returns:** Detections - A new Detections object created from YOLOv5 results.
- **Purpose:** Creates a Detections instance from YOLOv5 inference results.

#### `from_ultralytics`
```python
@classmethod
def from_ultralytics(cls, ultralytics_results) -> Detections
```
- **Parameters:**
  - `ultralytics_results`: The output results from Ultralytics models.
  
- **Returns:** Detections - A new Detections object created from Ultralytics results.
- **Purpose:** Creates a Detections instance from Ultralytics inference results.

#### `from_yolo_nas`
```python
@classmethod
def from_yolo_nas(cls, yolo_nas_results) -> Detections
```
- **Parameters:**
  - `yolo_nas_results`: The output results from YOLO NAS models.
  
- **Returns:** Detections - A new Detections object created from YOLO NAS results.
- **Purpose:** Creates a Detections instance from YOLO NAS inference results.

#### `from_tensorflow`
```python
@classmethod
def from_tensorflow(cls, tensorflow_results: dict, resolution_wh: tuple) -> Detections
```
- **Parameters:**
  - `tensorflow_results` (dict): The output results from TensorFlow models.
  - `resolution_wh` (tuple): The resolution of the input image (width, height).
  
- **Returns:** Detections - A new Detections object created from TensorFlow results.
- **Purpose:** Creates a Detections instance from TensorFlow inference results.

#### `from_deepsparse`
```python
@classmethod
def from_deepsparse(cls, deepsparse_results) -> Detections
```
- **Parameters:**
  - `deepsparse_results`: The output results from DeepSparse models.
  
- **Returns:** Detections - A new Detections object created from DeepSparse results.
- **Purpose:** Creates a Detections instance from DeepSparse inference results.

#### `from_mmdetection`
```python
@classmethod
def from_mmdetection(cls, mmdet_results) -> Detections
```
- **Parameters:**
  - `mmdet_results`: The output results from MMDetection models.
  
- **Returns:** Detections - A new Detections object created from MMDetection results.
- **Purpose:** Creates a Detections instance from MMDetection inference results.

#### `from_transformers`
```python
@classmethod
def from_transformers(cls, transformers_results: dict, id2label: Optional[Dict[int, str]] = None) -> Detections
```
- **Parameters:**
  - `transformers_results` (dict): The output results from Transformers models.
  - `id2label` (Optional[Dict[int, str]]): Mapping from class IDs to class names.
  
- **Returns:** Detections - A new Detections object created from Transformers results.
- **Purpose:** Creates a Detections instance from Transformers inference results.

#### `from_detectron2`
```python
@classmethod
def from_detectron2(cls, detectron2_results: Any) -> Detections
```
- **Parameters:**
  - `detectron2_results`: The output results from Detectron2 models.
  
- **Returns:** Detections - A new Detections object created from Detectron2 results.
- **Purpose:** Creates a Detections instance from Detectron2 inference results.

#### `from_inference`
```python
@classmethod
def from_inference(cls, roboflow_result: Union[dict, Any]) -> Detections
```
- **Parameters:**
  - `roboflow_result`: The output results from Roboflow inference.
  
- **Returns:** Detections - A new Detections object created from Roboflow results.
- **Purpose:** Creates a Detections instance from Roboflow inference results.

#### `from_sam`
```python
@classmethod
def from_sam(cls, sam_result: List[dict]) -> Detections
```
- **Parameters:**
  - `sam_result` (List[dict]): The output results from SAM models.
  
- **Returns:** Detections - A new Detections object created from SAM results.
- **Purpose:** Creates a Detections instance from SAM inference results.

#### `from_azure_analyze_image`
```python
@classmethod
def from_azure_analyze_image(cls, azure_result: dict, class_map: Optional[Dict[int, str]] = None) -> Detections
```
- **Parameters:**
  - `azure_result` (dict): The output results from Azure Analyze Image API.
  - `class_map` (Optional[Dict[int, str]]): Mapping from class IDs to class names.
  
- **Returns:** Detections - A new Detections object created from Azure results.
- **Purpose:** Creates a Detections instance from Azure Analyze Image results.

#### `from_paddledet`
```python
@classmethod
def from_paddledet(cls, paddledet_result) -> Detections
```
- **Parameters:**
  - `paddledet_result`: The output results from PaddleDetection models.
  
- **Returns:** Detections - A new Detections object created from PaddleDetection results.
- **Purpose:** Creates a Detections instance from PaddleDetection inference results.

#### `from_lmm`
```python
@classmethod
def from_lmm(cls, lmm: Union[LMM, str], result: Union[str, dict], **kwargs: Any) -> Detections
```
- **Parameters:**
  - `lmm` (Union[LMM, str]): The language model to use.
  - `result` (Union[str, dict]): The result from the language model.
  
- **Returns:** Detections - A new Detections object created from LMM results.
- **Purpose:** Creates a Detections instance from LMM inference results (deprecated).

#### `from_vlm`
```python
@classmethod
def from_vlm(cls, vlm: Union[VLM, str], result: Union[str, dict], **kwargs: Any) -> Detections
```
- **Parameters:**
  - `vlm` (Union[VLM, str]): The vision language model to use.
  - `result` (Union[str, dict]): The result from the vision language model.
  
- **Returns:** Detections - A new Detections object created from VLM results.
- **Purpose:** Creates a Detections instance from VLM inference results.

#### `from_easyocr`
```python
@classmethod
def from_easyocr(cls, easyocr_results: list) -> Detections
```
- **Parameters:**
  - `easyocr_results` (list): The output results from EasyOCR.
  
- **Returns:** Detections - A new Detections object created from EasyOCR results.
- **Purpose:** Creates a Detections instance from EasyOCR inference results.

#### `from_ncnn`
```python
@classmethod
def from_ncnn(cls, ncnn_results) -> Detections
```
- **Parameters:**
  - `ncnn_results`: The output results from NCNN models.
  
- **Returns:** Detections - A new Detections object created from NCNN results.
- **Purpose:** Creates a Detections instance from NCNN inference results.

#### `empty`
```python
@classmethod
def empty(cls) -> Detections
```
- **Returns:** Detections - An empty Detections object.
- **Purpose:** Creates and returns an empty Detections instance.

#### `is_empty`
```python
def is_empty(self) -> bool
```
- **Returns:** bool - True if the Detections object is empty, False otherwise.
- **Purpose:** Checks if the Detections object is empty.

#### `merge`
```python
@classmethod
def merge(cls, detections_list: List[Detections]) -> Detections
```
- **Parameters:**
  - `detections_list` (List[Detections]): A list of Detections objects to merge.
  
- **Returns:** Detections - A new Detections object containing merged detections.
- **Purpose:** Merges multiple Detections objects into one.

#### `get_anchors_coordinates`
```python
def get_anchors_coordinates(self, anchor: Position) -> np.ndarray
```
- **Parameters:**
  - `anchor` (Position): The position type to get anchor coordinates.
  
- **Returns:** np.ndarray - The coordinates of the specified anchor positions.
- **Purpose:** Retrieves the coordinates of specified anchor positions for detections.

#### `area`
```python
@property
def area(self) -> np.ndarray
```
- **Returns:** np.ndarray - The area of each detection (based on masks if available).
- **Purpose:** Calculates the area of each detection.

#### `box_area`
```python
@property
def box_area(self) -> np.ndarray
```
- **Returns:** np.ndarray - The area of each bounding box.
- **Purpose:** Calculates the area of each bounding box.

#### `with_nms`
```python
def with_nms(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections
```
- **Parameters:**
  - `threshold` (float): The IoU threshold for non-maximum suppression (default is 0.5).
  - `class_agnostic` (bool): If True, performs class-agnostic NMS (default is False).
  
- **Returns:** Detections - A new Detections object after applying NMS.
- **Purpose:** Applies non-maximum suppression to filter detections.

#### `with_nmm`
```python
def with_nmm(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections
```
- **Parameters:**
  - `threshold` (float): The IoU threshold for non-maximum merging (default is 0.5).
  - `class_agnostic` (bool): If True, performs class-agnostic NMM (default is False).
  
- **Returns:** Detections - A new Detections object after applying NMM.
- **Purpose:** Applies non-maximum merging to combine overlapping detections.