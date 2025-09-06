# 接口文档：RealESRGANer 类

## 1. 类：RealESRGANer

### 1.1 初始化方法：`__init__`

#### 函数名
`__init__`

#### 参数说明
- `scale` (int): 超分辨率缩放因子，通常为2或4。
- `model_path` (str): 预训练模型的路径，可以是URL（会自动下载）。
- `dni_weight` (list, optional): 深度网络插值权重，默认为None。
- `model` (nn.Module, optional): 定义的网络，默认为None。
- `tile` (int, optional): 图像切片大小，默认为0（不使用切片）。
- `tile_pad` (int, optional): 每个切片的填充大小，默认为10。
- `pre_pad` (int, optional): 输入图像的填充大小，默认为10。
- `half` (bool, optional): 是否在推理过程中使用半精度，默认为False。
- `device` (torch.device, optional): 指定设备，默认为None。
- `gpu_id` (int, optional): 指定GPU ID，默认为None。

#### 返回值说明
无返回值。

### 1.2 属性
- `scale`: 超分辨率缩放因子。
- `tile_size`: 切片大小。
- `tile_pad`: 切片填充大小。
- `pre_pad`: 输入图像填充大小。
- `mod_scale`: 模型缩放因子。
- `half`: 是否使用半精度。
- `device`: 设备信息。
- `model`: 加载的模型。

### 1.3 方法：`dni`

#### 函数名
`dni`

#### 参数说明
- `net_a` (str): 第一个网络的路径。
- `net_b` (str): 第二个网络的路径。
- `dni_weight` (list): 深度网络插值权重。
- `key` (str, optional): 权重字典的键，默认为'params'。
- `loc` (str, optional): 设备位置，默认为'cpu'。

#### 返回值说明
- 返回合并后的网络权重字典。

### 1.4 方法：`pre_process`

#### 函数名
`pre_process`

#### 参数说明
- `img` (numpy.ndarray): 输入图像，格式为HWC。

#### 返回值说明
无返回值。

### 1.5 方法：`process`

#### 函数名
`process`

#### 返回值说明
无返回值。

### 1.6 方法：`tile_process`

#### 函数名
`tile_process`

#### 返回值说明
无返回值。

### 1.7 方法：`post_process`

#### 函数名
`post_process`

#### 返回值说明
- 返回处理后的输出图像。

### 1.8 方法：`enhance`

#### 函数名
`enhance`

#### 参数说明
- `img` (numpy.ndarray): 输入图像，格式为HWC。
- `outscale` (float, optional): 输出缩放因子，默认为None。
- `alpha_upsampler` (str, optional): Alpha通道上采样方法，默认为'realesrgan'。

#### 返回值说明
- `output` (numpy.ndarray): 处理后的输出图像。
- `img_mode` (str): 输入图像的模式（'L', 'RGB', 'RGBA'）。

## 2. 调用示例

```python
import cv2
import numpy as np
from realesrgan import RealESRGANer

# 初始化RealESRGANer
model_path = 'path/to/pretrained/model.pth'
realesrgan = RealESRGANer(scale=4, model_path=model_path)

# 读取图像
img = cv2.imread('path/to/input/image.jpg')

# 增强图像
output_img, img_mode = realesrgan.enhance(img)

# 保存输出图像
cv2.imwrite('path/to/output/image.jpg', output_img)
```