# DeepFace.py 接口文档

## 类：Logger
### 初始化信息
- **Logger()**: 创建一个 Logger 实例，用于记录日志信息。

### 属性
- **log_level**: 日志级别，控制日志的输出级别。

### 方法
- **log(message: str, level: str = "INFO")**: 记录日志信息。
  - **参数**:
    - `message` (str): 要记录的日志信息。
    - `level` (str): 日志级别，默认为 "INFO"。
  - **返回值**: None
  - **范围**: 记录日志信息。

## 函数：build_model
### 接口说明
- **函数名**: build_model
- **参数**:
  - `model_name` (str): 模型标识符，支持多种人脸识别和属性分析模型。
  - `task` (str): 任务类型，默认为 "facial_recognition"。
- **返回值**: Any
- **范围**: 构建预训练模型。

### 调用示例
```python
model = build_model("VGG-Face", "facial_recognition")
```

## 函数：verify
### 接口说明
- **函数名**: verify
- **参数**:
  - `img1_path` (Union[str, np.ndarray, IO[bytes], List[float]): 第一个图像的路径或图像数据。
  - `img2_path` (Union[str, np.ndarray, IO[bytes], List[float]): 第二个图像的路径或图像数据。
  - `model_name` (str): 人脸识别模型，默认为 "VGG-Face"。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `distance_metric` (str): 相似度度量，默认为 "cosine"。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
  - `expand_percentage` (int): 扩展检测到的面部区域的百分比，默认为 0。
  - `normalization` (str): 输入图像的归一化方式，默认为 "base"。
  - `silent` (bool): 是否静默处理，默认为 False。
  - `threshold` (Optional[float]): 验证阈值，默认为 None。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
- **返回值**: Dict[str, Any]
- **范围**: 验证一对图像是否表示同一个人。

### 调用示例
```python
result = verify("path/to/image1.jpg", "path/to/image2.jpg")
```

## 函数：analyze
### 接口说明
- **函数名**: analyze
- **参数**:
  - `img_path` (Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]]): 图像路径或图像数据。
  - `actions` (Union[tuple, list]): 要分析的属性，默认为 ("emotion", "age", "gender", "race")。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
  - `expand_percentage` (int): 扩展检测到的面部区域的百分比，默认为 0。
  - `silent` (bool): 是否静默处理，默认为 False。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
- **返回值**: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]
- **范围**: 分析图像中的面部属性。

### 调用示例
```python
attributes = analyze("path/to/image.jpg")
```

## 函数：find
### 接口说明
- **函数名**: find
- **参数**:
  - `img_path` (Union[str, np.ndarray, IO[bytes]]): 图像路径或图像数据。
  - `db_path` (str): 数据库路径，包含图像文件。
  - `model_name` (str): 人脸识别模型，默认为 "VGG-Face"。
  - `distance_metric` (str): 相似度度量，默认为 "cosine"。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
  - `expand_percentage` (int): 扩展检测到的面部区域的百分比，默认为 0。
  - `threshold` (Optional[float]): 验证阈值，默认为 None。
  - `normalization` (str): 输入图像的归一化方式，默认为 "base"。
  - `silent` (bool): 是否静默处理，默认为 False。
  - `refresh_database` (bool): 是否刷新数据库，默认为 True。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
  - `batched` (bool): 是否批量处理，默认为 False。
- **返回值**: Union[List[pd.DataFrame], List[List[Dict[str, Any]]]
- **范围**: 在数据库中识别个体。

### 调用示例
```python
results = find("path/to/image.jpg", "path/to/database")
```

## 函数：represent
### 接口说明
- **函数名**: represent
- **参数**:
  - `img_path` (Union[str, np.ndarray, IO[bytes], Sequence[Union[str, np.ndarray, IO[bytes]]]): 图像路径或图像数据。
  - `model_name` (str): 人脸识别模型，默认为 "VGG-Face"。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
  - `expand_percentage` (int): 扩展检测到的面部区域的百分比，默认为 0。
  - `normalization` (str): 输入图像的归一化方式，默认为 "base"。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
  - `max_faces` (Optional[int]): 处理的面部数量限制，默认为 None。
- **返回值**: Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]
- **范围**: 将面部图像表示为多维向量嵌入。

### 调用示例
```python
embeddings = represent("path/to/image.jpg")
```

## 函数：stream
### 接口说明
- **函数名**: stream
- **参数**:
  - `db_path` (str): 数据库路径，包含图像文件。
  - `model_name` (str): 人脸识别模型，默认为 "VGG-Face"。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `distance_metric` (str): 相似度度量，默认为 "cosine"。
  - `enable_face_analysis` (bool): 是否启用面部分析，默认为 True。
  - `source` (Any): 视频流源，默认为 0（默认摄像头）。
  - `time_threshold` (int): 人脸识别的时间阈值，默认为 5。
  - `frame_threshold` (int): 人脸识别的帧阈值，默认为 5。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
  - `output_path` (Optional[str]): 输出视频保存路径，默认为 None。
  - `debug` (bool): 是否启用调试模式，默认为 False。
- **返回值**: None
- **范围**: 运行实时人脸识别和面部属性分析。

### 调用示例
```python
stream(db_path="path/to/database")
```

## 函数：extract_faces
### 接口说明
- **函数名**: extract_faces
- **参数**:
  - `img_path` (Union[str, np.ndarray, IO[bytes]]): 图像路径或图像数据。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
  - `expand_percentage` (int): 扩展检测到的面部区域的百分比，默认为 0。
  - `grayscale` (bool): 是否将输出面部图像转换为灰度，默认为 False。
  - `color_face` (str): 返回面部图像的颜色，默认为 "rgb"。
  - `normalize_face` (bool): 是否对输出面部图像进行归一化，默认为 True。
  - `anti_spoofing` (bool): 是否启用反欺诈检测，默认为 False。
- **返回值**: List[Dict[str, Any]]
- **范围**: 从给定图像中提取面部。

### 调用示例
```python
faces = extract_faces("path/to/image.jpg")
```

## 函数：cli
### 接口说明
- **函数名**: cli
- **参数**: None
- **返回值**: None
- **范围**: 提供命令行接口功能。

### 调用示例
```python
cli()
```

## 函数：detectFace (已弃用)
### 接口说明
- **函数名**: detectFace
- **参数**:
  - `img_path` (Union[str, np.ndarray]): 图像路径或图像数据。
  - `target_size` (tuple): 最终面部图像的形状，默认为 (224, 224)。
  - `detector_backend` (str): 人脸检测后端，默认为 "opencv"。
  - `enforce_detection` (bool): 是否强制检测人脸，默认为 True。
  - `align` (bool): 是否进行人脸对齐，默认为 True。
- **返回值**: Union[np.ndarray, None]
- **范围**: 检测面部图像（已弃用，建议使用 `extract_faces`）。

### 调用示例
```python
face = detectFace("path/to/image.jpg")
```

以上是 `DeepFace.py` 的接口文档，涵盖了所有类和函数的详细信息。