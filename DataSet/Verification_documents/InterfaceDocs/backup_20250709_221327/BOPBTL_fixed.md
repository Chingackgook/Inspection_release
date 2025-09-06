# 接口文档

## 类：`Mapping_Model`

### 初始化方法：`__init__(self, nc, mc=64, n_blocks=3, norm="instance", padding_type="reflect", opt=None)`

- **参数说明**：
  - `nc` (int): 输入通道数。
  - `mc` (int, optional): 最大通道数，默认为64。
  - `n_blocks` (int, optional): 残差块的数量，默认为3。
  - `norm` (str, optional): 归一化类型，默认为"instance"。
  - `padding_type` (str, optional): 填充类型，默认为"reflect"。
  - `opt` (optional): 其他选项。

- **返回值说明**：无返回值。

### 方法：`forward(self, input)`

- **参数说明**：
  - `input` (Tensor): 输入张量。

- **返回值说明**：
  - (Tensor): 经过模型处理后的输出张量。

---

## 类：`Pix2PixHDModel_Mapping`

### 方法：`name(self)`

- **参数说明**：无参数。

- **返回值说明**：
  - (str): 返回类的名称 "Pix2PixHDModel_Mapping"。

### 方法：`init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_smooth_l1, stage_1_feat_l2)`

- **参数说明**：
  - `use_gan_feat_loss` (bool): 是否使用GAN特征损失。
  - `use_vgg_loss` (bool): 是否使用VGG损失。
  - `use_smooth_l1` (bool): 是否使用平滑L1损失。
  - `stage_1_feat_l2` (bool): 第一阶段特征L2损失的标志。

- **返回值说明**：
  - (function): 返回一个损失过滤函数。

### 方法：`initialize(self, opt)`

- **参数说明**：
  - `opt` (object): 选项对象，包含模型初始化所需的参数。

- **返回值说明**：无返回值。

### 方法：`encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False)`

- **参数说明**：
  - `label_map` (Tensor): 标签图。
  - `inst_map` (Tensor, optional): 实例图，默认为None。
  - `real_image` (Tensor, optional): 真实图像，默认为None。
  - `feat_map` (Tensor, optional): 特征图，默认为None。
  - `infer` (bool, optional): 推理标志，默认为False。

- **返回值说明**：
  - (tuple): 返回编码后的输入标签、实例图、真实图像和特征图。

### 方法：`discriminate(self, input_label, test_image, use_pool=False)`

- **参数说明**：
  - `input_label` (Tensor): 输入标签。
  - `test_image` (Tensor): 测试图像。
  - `use_pool` (bool, optional): 是否使用图像池，默认为False。

- **返回值说明**：
  - (Tensor): 判别器的输出。

### 方法：`forward(self, label, inst, image, feat, pair=True, infer=False, last_label=None, last_image=None)`

- **参数说明**：
  - `label` (Tensor): 输入标签。
  - `inst` (Tensor): 实例图。
  - `image` (Tensor): 真实图像。
  - `feat` (Tensor): 特征图。
  - `pair` (bool, optional): 是否成对处理，默认为True。
  - `infer` (bool, optional): 推理标志，默认为False。
  - `last_label` (Tensor, optional): 上一个标签，默认为None。
  - `last_image` (Tensor, optional): 上一张图像，默认为None。

- **返回值说明**：
  - (list): 返回损失值和生成的图像（如果推理为True）。

### 方法：`inference(self, label, inst)`

- **参数说明**：
  - `label` (Tensor): 输入标签。
  - `inst` (Tensor): 实例图。

- **返回值说明**：
  - (Tensor): 生成的图像。

---

## 示例调用

```python
from options.test_options import TestOptions
from models.models import create_model
from models.mapping_model import Pix2PixHDModel_Mapping 

opt = TestOptions().parse(save=False)
parameter_set(opt)
model = Pix2PixHDModel_Mapping()
model.initialize(opt)
model.eval()
generated = model.inference(input, mask)
```