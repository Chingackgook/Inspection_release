# 接口文档: FaceAnalysis 类

## 1. 类：FaceAnalysis

### 1.1 初始化方法

#### 方法名
`__init__`

#### 参数说明
- `name` (str): 模型名称，默认为 `DEFAULT_MP_NAME`。
- `root` (str): 模型文件的根目录，默认为 `'~/.insightface'`。
- `allowed_modules` (list): 允许的模块列表，默认为 `None`。
- `**kwargs`: 额外的关键字参数。

#### 返回值说明
无返回值。

#### 范围说明
此方法用于初始化 FaceAnalysis 类，加载指定路径下的模型，并根据 allowed_modules 过滤掉不需要的模型。

---

### 1.2 属性

- `models` (dict): 存储加载的模型，键为任务名称，值为模型实例。
- `model_dir` (str): 模型文件的存储路径。
- `det_model` (模型实例): 检测模型，必须存在于模型中。

---

### 1.3 方法：prepare

#### 方法名
`prepare`

#### 参数说明
- `ctx_id` (int): 上下文ID，通常用于指定设备（如 GPU 设备）。
- `det_thresh` (float): 检测阈值，默认为 `0.5`。
- `det_size` (tuple): 检测尺寸，格式为 `(宽度, 高度)`，默认为 `(640, 640)`。

#### 返回值说明
无返回值。

#### 范围说明
此方法用于准备检测模型和其他模型的输入尺寸和阈值。

---

### 1.4 方法：get

#### 方法名
`get`

#### 参数说明
- `img` (ndarray): 输入的图像数据。
- `max_num` (int): 最大检测人数，默认为 `0`，表示不限制。
- `det_metric` (str): 检测的度量方式，默认为 `'default'`。

#### 返回值说明
- 返回一个列表，其中每个元素是一个 Face 对象，包含检测到的人脸的相关信息。

#### 范围说明
此方法用于从输入图像中检测人脸，返回检测到的人脸信息。

---

### 1.5 方法：draw_on

#### 方法名
`draw_on`

#### 参数说明
- `img` (ndarray): 输入的图像数据，用于绘制人脸框。
- `faces` (list): Face 对象的列表，将在图像上绘制。

#### 返回值说明
- 返回处理后的图像，其中人脸框和关键点已绘制。

#### 范围说明
此方法用于在输入图像上绘制检测到的人脸边框和关键点。

---

## 示例调用

```python
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
``` 

以上示例展示了如何创建 `FaceAnalysis` 实例、准备模型、获取人脸信息并在图像上绘制人脸框。