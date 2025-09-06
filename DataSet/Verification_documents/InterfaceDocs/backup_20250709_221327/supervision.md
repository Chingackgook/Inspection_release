# 接口文档

## Detections 类

### 初始化方法
```python
def __init__(self, xyxy: np.ndarray, mask: Optional[np.ndarray] = None, confidence: Optional[np.ndarray] = None, class_id: Optional[np.ndarray] = None, tracker_id: Optional[np.ndarray] = None, data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict), metadata: Dict[str, Any] = field(default_factory=dict)
```
- **参数说明**:
  - `xyxy`: 形状为 (N, 4) 的 numpy 数组，表示 N 个检测框的坐标。
  - `mask`: 可选，形状为 (N, H, W) 的 numpy 数组，表示 N 个检测的掩码。
  - `confidence`: 可选，形状为 (N,) 的 numpy 数组，表示 N 个检测的置信度。
  - `class_id`: 可选，形状为 (N,) 的 numpy 数组，表示 N 个检测的类别 ID。
  - `tracker_id`: 可选，形状为 (N,) 的 numpy 数组，表示 N 个检测的跟踪 ID。
  - `data`: 可选，字典，包含额外的检测数据。
  - `metadata`: 可选，字典，包含元数据。

### 属性
- `xyxy`: 返回检测框的坐标。
- `mask`: 返回检测的掩码。
- `confidence`: 返回检测的置信度。
- `class_id`: 返回检测的类别 ID。
- `tracker_id`: 返回检测的跟踪 ID。
- `data`: 返回额外的检测数据。
- `metadata`: 返回元数据。
- `area`: 返回每个检测的面积。
- `box_area`: 返回每个检测框的面积。

### 方法
#### `__len__`
```python
def __len__(self) -> int
```
- **返回值**: 返回检测对象中的检测数量。
- **作用**: 获取检测数量。

#### `__iter__`
```python
def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[float], Optional[int], Optional[int], Dict[str, Union[np.ndarray, List]]]]
```
- **返回值**: 返回一个迭代器，迭代每个检测的元组。
- **作用**: 迭代检测对象，获取每个检测的详细信息。

#### `__eq__`
```python
def __eq__(self, other: Detections) -> bool
```
- **参数说明**:
  - `other`: 另一个 Detections 对象。
- **返回值**: 返回布尔值，表示两个 Detections 对象是否相等。
- **作用**: 比较两个 Detections 对象。

#### `from_yolov5`
```python
@classmethod
def from_yolov5(cls, yolov5_results) -> Detections
```
- **参数说明**:
  - `yolov5_results`: YOLOv5 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 YOLOv5 的推理结果创建 Detections 实例。

#### `from_ultralytics`
```python
@classmethod
def from_ultralytics(cls, ultralytics_results) -> Detections
```
- **参数说明**:
  - `ultralytics_results`: Ultralytics 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 Ultralytics 的推理结果创建 Detections 实例。

#### `from_yolo_nas`
```python
@classmethod
def from_yolo_nas(cls, yolo_nas_results) -> Detections
```
- **参数说明**:
  - `yolo_nas_results`: YOLO NAS 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 YOLO NAS 的推理结果创建 Detections 实例。

#### `from_tensorflow`
```python
@classmethod
def from_tensorflow(cls, tensorflow_results: dict, resolution_wh: tuple) -> Detections
```
- **参数说明**:
  - `tensorflow_results`: TensorFlow 模型的推理结果。
  - `resolution_wh`: 图像的分辨率 (宽, 高)。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 TensorFlow 的推理结果创建 Detections 实例。

#### `from_deepsparse`
```python
@classmethod
def from_deepsparse(cls, deepsparse_results) -> Detections
```
- **参数说明**:
  - `deepsparse_results`: DeepSparse 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 DeepSparse 的推理结果创建 Detections 实例。

#### `from_mmdetection`
```python
@classmethod
def from_mmdetection(cls, mmdet_results) -> Detections
```
- **参数说明**:
  - `mmdet_results`: MMDetection 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 MMDetection 的推理结果创建 Detections 实例。

#### `from_transformers`
```python
@classmethod
def from_transformers(cls, transformers_results: dict, id2label: Optional[Dict[int, str]] = None) -> Detections
```
- **参数说明**:
  - `transformers_results`: Transformers 模型的推理结果。
  - `id2label`: 可选，类别 ID 到标签的映射。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 Transformers 的推理结果创建 Detections 实例。

#### `from_detectron2`
```python
@classmethod
def from_detectron2(cls, detectron2_results: Any) -> Detections
```
- **参数说明**:
  - `detectron2_results`: Detectron2 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 Detectron2 的推理结果创建 Detections 实例。

#### `from_inference`
```python
@classmethod
def from_inference(cls, roboflow_result: Union[dict, Any]) -> Detections
```
- **参数说明**:
  - `roboflow_result`: Roboflow 的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 Roboflow 的推理结果创建 Detections 实例。

#### `from_sam`
```python
@classmethod
def from_sam(cls, sam_result: List[dict]) -> Detections
```
- **参数说明**:
  - `sam_result`: SAM 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 SAM 的推理结果创建 Detections 实例。

#### `from_azure_analyze_image`
```python
@classmethod
def from_azure_analyze_image(cls, azure_result: dict, class_map: Optional[Dict[int, str]] = None) -> Detections
```
- **参数说明**:
  - `azure_result`: Azure API 的推理结果。
  - `class_map`: 可选，类别 ID 到名称的映射。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 Azure 的推理结果创建 Detections 实例。

#### `from_paddledet`
```python
@classmethod
def from_paddledet(cls, paddledet_result) -> Detections
```
- **参数说明**:
  - `paddledet_result`: PaddleDetection 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 PaddleDetection 的推理结果创建 Detections 实例。

#### `from_lmm`
```python
@classmethod
def from_lmm(cls, lmm: Union[LMM, str], result: Union[str, dict], **kwargs: Any) -> Detections
```
- **参数说明**:
  - `lmm`: LMM 枚举或字符串。
  - `result`: 推理结果。
  - `**kwargs`: 其他参数。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 LMM 的推理结果创建 Detections 实例（已弃用，建议使用 `from_vlm`）。

#### `from_vlm`
```python
@classmethod
def from_vlm(cls, vlm: Union[VLM, str], result: Union[str, dict], **kwargs: Any) -> Detections
```
- **参数说明**:
  - `vlm`: VLM 枚举或字符串。
  - `result`: 推理结果。
  - `**kwargs`: 其他参数。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 VLM 的推理结果创建 Detections 实例。

#### `from_easyocr`
```python
@classmethod
def from_easyocr(cls, easyocr_results: list) -> Detections
```
- **参数说明**:
  - `easyocr_results`: EasyOCR 的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 EasyOCR 的推理结果创建 Detections 实例。

#### `from_ncnn`
```python
@classmethod
def from_ncnn(cls, ncnn_results) -> Detections
```
- **参数说明**:
  - `ncnn_results`: NCNN 模型的推理结果。
- **返回值**: 返回一个 Detections 对象。
- **作用**: 从 NCNN 的推理结果创建 Detections 实例。

#### `empty`
```python
@classmethod
def empty(cls) -> Detections
```
- **返回值**: 返回一个空的 Detections 对象。
- **作用**: 创建一个空的 Detections 实例。

#### `is_empty`
```python
def is_empty(self) -> bool
```
- **返回值**: 返回布尔值，表示 Detections 对象是否为空。
- **作用**: 检查 Detections 对象是否为空。

#### `merge`
```python
@classmethod
def merge(cls, detections_list: List[Detections]) -> Detections
```
- **参数说明**:
  - `detections_list`: Detections 对象的列表。
- **返回值**: 返回合并后的 Detections 对象。
- **作用**: 合并多个 Detections 对象。

#### `get_anchors_coordinates`
```python
def get_anchors_coordinates(self, anchor: Position) -> np.ndarray
```
- **参数说明**:
  - `anchor`: 位置枚举，指定锚点位置。
- **返回值**: 返回锚点坐标的 numpy 数组。
- **作用**: 获取指定锚点位置的坐标。

#### `area`
```python
@property
def area(self) -> np.ndarray
```
- **返回值**: 返回每个检测的面积。
- **作用**: 计算并返回每个检测的面积。

#### `box_area`
```python
@property
def box_area(self) -> np.ndarray
```
- **返回值**: 返回每个检测框的面积。
- **作用**: 计算并返回每个检测框的面积。

#### `with_nms`
```python
def with_nms(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections
```
- **参数说明**:
  - `threshold`: NMS 阈值，默认为 0.5。
  - `class_agnostic`: 是否进行类别无关的 NMS，默认为 False。
- **返回值**: 返回经过 NMS 处理后的 Detections 对象。
- **作用**: 执行非极大值抑制（NMS）。

#### `with_nmm`
```python
def with_nmm(self, threshold: float = 0.5, class_agnostic: bool = False) -> Detections
```
- **参数说明**:
  - `threshold`: NMM 阈值，默认为 0.5。
  - `class_agnostic`: 是否进行类别无关的 NMM，默认为 False。
- **返回值**: 返回经过 NMM 处理后的 Detections 对象。
- **作用**: 执行非极大值合并（NMM）。

#### `stack_or_none`
```python
def stack_or_none(name: str)
```
- **参数说明**:
  - `name`: 字符串，表示要堆叠的字段名称。
- **返回值**: 返回堆叠后的 numpy 数组或 None。
- **作用**: 堆叠指定字段的值，如果所有值均为 None，则返回 None。