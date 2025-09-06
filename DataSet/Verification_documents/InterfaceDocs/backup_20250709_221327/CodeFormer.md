# 接口文档

## 类: FaceRestoreHelper
### 描述
封装从原始图像到最终修复的完整处理流程，实现了人脸修复流程的全链路管理，通过标准化的人脸检测-对齐-修复-逆向融合流程，结合多模型协作（检测/解析/超分）和自适应参数配置，完成高质量的人脸修复与自然融合。

### 属性
- `upscale_factor`: (int) 放大因子，影响输出图像的分辨率。
- `face_size`: (tuple) 人脸图像的大小，默认为(512, 512)。
- `crop_ratio`: (tuple) 裁剪比例，默认为(1, 1)。
- `det_model`: (str) 人脸检测模型的名称，默认为'retinaface_resnet50'。
- `save_ext`: (str) 保存图像的扩展名，默认为'png'。
- `template_3points`: (bool) 是否使用三点模板，默认为False。
- `pad_blur`: (bool) 是否对图像进行模糊填充，默认为False。
- `use_parse`: (bool) 是否使用解析模型，默认为False。
- `device`: (torch.device) 设备类型，默认为None，自动选择可用设备。
- `all_landmarks_5`: (list) 存储检测到的5个关键点的列表。
- `det_faces`: (list) 存储检测到的人脸区域。
- `affine_matrices`: (list) 存储仿射变换矩阵。
- `inverse_affine_matrices`: (list) 存储逆仿射变换矩阵。
- `cropped_faces`: (list) 存储裁剪后的人脸图像。
- `restored_faces`: (list) 存储修复后的人脸图像。
- `pad_input_imgs`: (list) 存储填充后的输入图像。

### 方法

#### `__init__(self, upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', template_3points=False, pad_blur=False, use_parse=False, device=None)`
- **参数**:
  - `upscale_factor`: (int) 放大因子。
  - `face_size`: (int) 人脸图像的大小，默认为512。
  - `crop_ratio`: (tuple) 裁剪比例，默认为(1, 1)。
  - `det_model`: (str) 人脸检测模型的名称，默认为'retinaface_resnet50'。
  - `save_ext`: (str) 保存图像的扩展名，默认为'png'。
  - `template_3points`: (bool) 是否使用三点模板，默认为False。
  - `pad_blur`: (bool) 是否对图像进行模糊填充，默认为False。
  - `use_parse`: (bool) 是否使用解析模型，默认为False。
  - `device`: (torch.device) 设备类型，默认为None。
- **返回值**: None
- **范围**: `upscale_factor` 应为正整数，`face_size` 应为正整数，`crop_ratio` 应为大于等于1的元组。

#### `set_upscale_factor(self, upscale_factor)`
- **参数**:
  - `upscale_factor`: (int) 新的放大因子。
- **返回值**: None
- **范围**: `upscale_factor` 应为正整数。

#### `read_image(self, img)`
- **参数**:
  - `img`: (str or np.ndarray) 图像路径或已加载的图像。
- **返回值**: None
- **范围**: 图像应为有效的图像格式，支持16位和8位图像。

#### `init_dlib(self, detection_path, landmark5_path)`
- **参数**:
  - `detection_path`: (str) 人脸检测模型的路径。
  - `landmark5_path`: (str) 5个关键点预测模型的路径。
- **返回值**: (face_detector, shape_predictor_5) 人脸检测器和形状预测器。
- **范围**: 模型路径应为有效的文件路径。

#### `get_face_landmarks_5_dlib(self, only_keep_largest=False, scale=1)`
- **参数**:
  - `only_keep_largest`: (bool) 是否仅保留最大的检测到的人脸，默认为False。
  - `scale`: (float) 缩放因子，默认为1。
- **返回值**: (int) 检测到的人脸数量。
- **范围**: `scale` 应为正数。

#### `get_face_landmarks_5(self, only_keep_largest=False, only_center_face=False, resize=None, blur_ratio=0.01, eye_dist_threshold=None)`
- **参数**:
  - `only_keep_largest`: (bool) 是否仅保留最大的检测到的人脸，默认为False。
  - `only_center_face`: (bool) 是否仅保留中心的人脸，默认为False。
  - `resize`: (int or None) 重新调整图像大小，默认为None。
  - `blur_ratio`: (float) 模糊比例，默认为0.01。
  - `eye_dist_threshold`: (float or None) 眼睛距离阈值，默认为None。
- **返回值**: (int) 检测到的人脸数量。
- **范围**: `resize` 应为正整数，`blur_ratio` 应为非负数，`eye_dist_threshold` 应为非负数或None。

#### `align_warp_face(self, save_cropped_path=None, border_mode='constant')`
- **参数**:
  - `save_cropped_path`: (str or None) 保存裁剪后人脸的路径，默认为None。
  - `border_mode`: (str) 边界模式，默认为'constant'。
- **返回值**: None
- **范围**: `border_mode` 应为有效的边界模式字符串。

#### `get_inverse_affine(self, save_inverse_affine_path=None)`
- **参数**:
  - `save_inverse_affine_path`: (str or None) 保存逆仿射矩阵的路径，默认为None。
- **返回值**: None
- **范围**: `save_inverse_affine_path` 应为有效的文件路径或None。

#### `add_restored_face(self, restored_face, input_face=None)`
- **参数**:
  - `restored_face`: (np.ndarray) 修复后的人脸图像。
  - `input_face`: (np.ndarray or None) 输入的人脸图像，默认为None。
- **返回值**: None
- **范围**: `restored_face` 应为有效的图像格式。

#### `paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None)`
- **参数**:
  - `save_path`: (str or None) 保存合成图像的路径，默认为None。
  - `upsample_img`: (np.ndarray or None) 可选的上采样图像，默认为None。
  - `draw_box`: (bool) 是否绘制边框，默认为False。
  - `face_upsampler`: (object or None) 人脸上采样器，默认为None。
- **返回值**: (np.ndarray) 合成后的图像。
- **范围**: `save_path` 应为有效的文件路径或None，`upsample_img` 应为有效的图像格式或None。

#### `clean_all(self)`
- **参数**: None
- **返回值**: None
- **范围**: None

以上是 `FaceRestoreHelper` 类及其方法的接口文档，详细描述了每个方法的参数、返回值和范围。