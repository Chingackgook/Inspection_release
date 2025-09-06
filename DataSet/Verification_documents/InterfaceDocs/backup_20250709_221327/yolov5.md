# 接口文档

## 类：`DetectMultiBackend`

### 初始化方法：`__init__`
- **函数名**: `__init__`
- **参数说明**:
  - `weights` (str or list): 模型权重文件路径，支持多种格式（如 .pt, .onnx, .tflite 等）。
  - `device` (torch.device): 设备类型，默认为 CPU。
  - `dnn` (bool): 是否使用 OpenCV DNN，默认为 False。
  - `data` (str or dict): 数据集配置文件路径或字典，包含类名等信息。
  - `fp16` (bool): 是否使用 FP16 精度，默认为 False。
  - `fuse` (bool): 是否融合模型，默认为 True。
- **返回值说明**: 无返回值。
- **范围说明**: 用于初始化模型并加载指定格式的权重文件。

### 属性
- `model`: 加载的模型实例。
- `device`: 当前使用的设备。
- `fp16`: 是否使用 FP16 精度。
- `nhwc`: 是否使用 NHWC 格式。
- `stride`: 模型的步幅。
- `names`: 类别名称列表。

### 方法：`forward`
- **函数名**: `forward`
- **参数说明**:
  - `im` (torch.Tensor): 输入图像，形状为 (batch, channel, height, width)。
  - `augment` (bool): 是否进行数据增强，默认为 False。
  - `visualize` (bool): 是否可视化推理过程，默认为 False。
- **返回值说明**: 返回推理结果，类型为 torch.Tensor 或 list。
- **范围说明**: 执行模型推理，处理输入图像并返回结果。

### 方法：`from_numpy`
- **函数名**: `from_numpy`
- **参数说明**:
  - `x` (np.ndarray): 输入的 NumPy 数组。
- **返回值说明**: 返回转换后的 torch.Tensor。
- **范围说明**: 将 NumPy 数组转换为 PyTorch 张量。

### 方法：`warmup`
- **函数名**: `warmup`
- **参数说明**:
  - `imgsz` (tuple): 输入图像的尺寸，默认为 (1, 3, 640, 640)。
- **返回值说明**: 无返回值。
- **范围说明**: 预热模型，以提高推理速度。

### 静态方法：`_model_type`
- **函数名**: `_model_type`
- **参数说明**:
  - `p` (str): 模型文件路径。
- **返回值说明**: 返回一个布尔值列表，指示支持的模型类型。
- **范围说明**: 确定模型文件的类型。

### 静态方法：`_load_metadata`
- **函数名**: `_load_metadata`
- **参数说明**:
  - `f` (Path): 元数据文件路径。
- **返回值说明**: 返回步幅和类名列表。
- **范围说明**: 加载模型的元数据。

---

## 示例调用

```python
import torch

# 加载 YOLOv5 模型（选项：yolov5n, yolov5s, yolov5m, yolov5l, yolov5x）
model = DetectMultiBackend(weights="yolov5s.pt", device=torch.device("cuda:0"))

# 定义输入图像源（URL、本地文件、PIL 图像、OpenCV 帧、numpy 数组或列表）
img = "data/images/bus.jpg"  # 示例图像

# 执行推理（自动处理批处理、调整大小、归一化）
results = model.forward(torch.from_numpy(np.array(img)))

# 处理结果（选项：.print(), .show(), .save(), .crop(), .pandas()）
results.print()  # 将结果打印到控制台
results.show()  # 在窗口中显示结果
# results.save()  # 将结果保存到 runs/detect/exp
```