# 接口文档

## 类：GFPGANer

### 初始化方法：`__init__`

#### 参数说明：
- `model_path` (str): GFPGAN模型的路径，可以是URL（会自动下载）。
- `upscale` (float): 最终输出的放大倍数。默认值：2。
- `arch` (str): GFPGAN架构。选项：clean | original。默认值：clean。
- `channel_multiplier` (int): StyleGAN2大网络的通道倍增器。默认值：2。
- `bg_upsampler` (nn.Module): 背景的上采样器。默认值：None。
- `device` (torch.device, optional): 设备类型，默认为自动选择（CUDA或CPU）。

#### 返回值说明：
无返回值。

---

### 方法：`enhance`

#### 参数说明：
- `img` (numpy.ndarray): 输入图像，格式为BGR。
- `has_aligned` (bool): 输入图像是否已经对齐。默认值：False。
- `only_center_face` (bool): 是否仅处理中心人脸。默认值：False。
- `paste_back` (bool): 是否将修复后的人脸粘贴回原图。默认值：True。
- `weight` (float): 修复人脸时的权重。默认值：0.5。

#### 返回值说明：
- `cropped_faces` (list): 裁剪的人脸图像列表。
- `restored_faces` (list): 修复后的人脸图像列表。
- `restored_img` (numpy.ndarray or None): 修复后粘贴回原图的图像，如果`paste_back`为False则返回None。

#### 范围说明：
- `img`的尺寸应为任意大小，但建议为高质量图像。
- `weight`的值应在0到1之间。

#### 调用示例：
```python
gfpganer = GFPGANer(model_path='path/to/model.pth')
cropped_faces, restored_faces, restored_img = gfpganer.enhance(img, has_aligned=False)
```

---

## 函数：`load_file_from_url`

#### 参数说明：
- `url` (str): 文件的URL地址。
- `model_dir` (str): 下载文件的目录。
- `progress` (bool): 是否显示下载进度。默认值：True。
- `file_name` (str or None): 保存的文件名，默认为None。

#### 返回值说明：
- (str): 下载文件的本地路径。

#### 范围说明：
- `url`应为有效的文件下载链接。

#### 调用示例：
```python
local_path = load_file_from_url('https://example.com/model.pth', model_dir='./models')
```

---

## 函数：`img2tensor`

#### 参数说明：
- `img` (numpy.ndarray): 输入图像，格式为HWC。
- `bgr2rgb` (bool): 是否将BGR格式转换为RGB格式。默认值：True。
- `float32` (bool): 是否将图像转换为float32类型。默认值：True。

#### 返回值说明：
- (torch.Tensor): 转换后的图像张量。

#### 范围说明：
- `img`的尺寸应为(H, W, C)。

#### 调用示例：
```python
tensor = img2tensor(img, bgr2rgb=True)
```

---

## 函数：`normalize`

#### 参数说明：
- `tensor` (torch.Tensor): 输入的图像张量。
- `mean` (tuple): 均值，用于归一化。
- `std` (tuple): 标准差，用于归一化。
- `inplace` (bool): 是否在原地进行归一化。默认值：False。

#### 返回值说明：
无返回值。

#### 范围说明：
- `mean`和`std`应为长度为3的元组，分别对应RGB通道。

#### 调用示例：
```python
normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

---

## 函数：`tensor2img`

#### 参数说明：
- `tensor` (torch.Tensor): 输入的图像张量。
- `rgb2bgr` (bool): 是否将RGB格式转换为BGR格式。默认值：True。
- `min_max` (tuple): 图像的最小值和最大值。默认值：(-1, 1)。

#### 返回值说明：
- (numpy.ndarray): 转换后的图像，格式为HWC。

#### 范围说明：
- `tensor`的尺寸应为(C, H, W)。

#### 调用示例：
```python
img = tensor2img(tensor, rgb2bgr=True)
```

---

## 函数：`FaceRestoreHelper`

### 初始化方法：`__init__`

#### 参数说明：
- `upscale` (float): 最终输出的放大倍数。
- `face_size` (int): 人脸图像的尺寸。默认值：512。
- `crop_ratio` (tuple): 裁剪比例。默认值：(1, 1)。
- `det_model` (str): 人脸检测模型。默认值：'retinaface_resnet50'。
- `save_ext` (str): 保存的文件扩展名。默认值：'png'。
- `use_parse` (bool): 是否使用解析。默认值：True。
- `device` (torch.device): 设备类型。
- `model_rootpath` (str): 模型根路径。

#### 返回值说明：
无返回值。

---

### 方法：`clean_all`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

---

### 方法：`read_image`

#### 参数说明：
- `img` (numpy.ndarray): 输入图像，格式为BGR。

#### 返回值说明：
无返回值。

---

### 方法：`get_face_landmarks_5`

#### 参数说明：
- `only_center_face` (bool): 是否仅处理中心人脸。默认值：False。
- `eye_dist_threshold` (int): 眼睛距离阈值。默认值：5。

#### 返回值说明：
无返回值。

---

### 方法：`align_warp_face`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

---

### 方法：`add_restored_face`

#### 参数说明：
- `restored_face` (numpy.ndarray): 修复后的人脸图像。

#### 返回值说明：
无返回值。

---

### 方法：`get_inverse_affine`

#### 参数说明：
- `None`: 无参数。

#### 返回值说明：
无返回值。

---

### 方法：`paste_faces_to_input_image`

#### 参数说明：
- `upsample_img` (numpy.ndarray or None): 上采样的背景图像，如果没有则为None。

#### 返回值说明：
- (numpy.ndarray): 粘贴了修复人脸的图像。

---

以上是GFPGANer类及其相关函数的接口文档，涵盖了初始化信息、属性、方法及其参数和返回值说明。