以下是根据提供的接口实现信息生成的接口文档：

### 1. `pre_check`
- **函数名**: `pre_check`
- **参数**: 无
- **返回值**: `bool` - 返回 `True` 表示预检查通过，返回 `False` 表示预检查失败。
- **范围**: 无
- **作用简述**: 检查所需的模型文件是否存在，并下载缺失的文件。

---

### 2. `pre_start`
- **函数名**: `pre_start`
- **参数**: 无
- **返回值**: `bool` - 返回 `True` 表示启动检查通过，返回 `False` 表示启动检查失败。
- **范围**: 无
- **作用简述**: 检查源路径和目标路径的有效性，确保至少有一个人脸被检测到。

---

### 3. `get_face_swapper`
- **函数名**: `get_face_swapper`
- **参数**: 无
- **返回值**: `Any` - 返回人脸交换模型的实例。
- **范围**: 无
- **作用简述**: 获取人脸交换模型的实例，确保模型只被加载一次。

---

### 4. `swap_face`
- **函数名**: `swap_face`
- **参数**:
  - `source_face: Face` - 源人脸对象。
  - `target_face: Face` - 目标人脸对象。
  - `temp_frame: Frame` - 输入图像帧。
- **返回值**: `Frame` - 返回处理后的人脸交换图像帧。
- **范围**: 无
- **作用简述**: 在给定的图像帧中执行人脸交换操作。

---

### 5. `process_frame`
- **函数名**: `process_frame`
- **参数**:
  - `source_face: Face` - 源人脸对象。
  - `temp_frame: Frame` - 输入图像帧。
- **返回值**: `Frame` - 返回处理后的人脸交换图像帧。
- **范围**: 无
- **作用简述**: 处理单个图像帧，检测目标人脸并执行人脸交换。

---

### 6. `process_frame_v2`
- **函数名**: `process_frame_v2`
- **参数**:
  - `temp_frame: Frame` - 输入图像帧。
  - `temp_frame_path: str` - 输入图像帧的路径（可选）。
- **返回值**: `Frame` - 返回处理后的人脸交换图像帧。
- **范围**: 无
- **作用简述**: 处理图像帧，支持多张人脸的交换，适用于不同的输入类型（图像或视频）。

---

### 7. `process_frames`
- **函数名**: `process_frames`
- **参数**:
  - `source_path: str` - 源图像路径。
  - `temp_frame_paths: List[str]` - 目标图像帧路径列表。
  - `progress: Any` - 进度更新对象（可选）。
- **返回值**: `None`
- **范围**: 无
- **作用简述**: 处理多张图像帧，执行人脸交换并保存结果。

---

### 8. `process_image`
- **函数名**: `process_image`
- **参数**:
  - `source_path: str` - 源图像路径。
  - `target_path: str` - 目标图像路径。
  - `output_path: str` - 输出图像路径。
- **返回值**: `None`
- **范围**: 无
- **作用简述**: 处理单张图像，执行人脸交换并保存结果。

---

### 9. `process_video`
- **函数名**: `process_video`
- **参数**:
  - `source_path: str` - 源视频路径。
  - `temp_frame_paths: List[str]` - 目标视频帧路径列表。
- **返回值**: `None`
- **范围**: 无
- **作用简述**: 处理视频文件，提取帧并执行人脸交换。

---

### 10. `create_lower_mouth_mask`
- **函数名**: `create_lower_mouth_mask`
- **参数**:
  - `face: Face` - 人脸对象。
  - `frame: Frame` - 输入图像帧。
- **返回值**: `(np.ndarray, np.ndarray, tuple, np.ndarray)` - 返回口罩、口部切割、边界框和下唇多边形。
- **范围**: 无
- **作用简述**: 创建下嘴唇的掩膜，用于后续的嘴部区域处理。

---

### 11. `draw_mouth_mask_visualization`
- **函数名**: `draw_mouth_mask_visualization`
- **参数**:
  - `frame: Frame` - 输入图像帧。
  - `face: Face` - 人脸对象。
  - `mouth_mask_data: tuple` - 包含口罩数据的元组。
- **返回值**: `Frame` - 返回可视化后的图像帧。
- **范围**: 无
- **作用简述**: 在图像帧上绘制嘴部掩膜的可视化效果。

---

### 12. `apply_mouth_area`
- **函数名**: `apply_mouth_area`
- **参数**:
  - `frame: np.ndarray` - 输入图像帧。
  - `mouth_cutout: np.ndarray` - 嘴部切割图像。
  - `mouth_box: tuple` - 嘴部区域的边界框。
  - `face_mask: np.ndarray` - 人脸掩膜。
  - `mouth_polygon: np.ndarray` - 嘴部多边形。
- **返回值**: `np.ndarray` - 返回处理后的图像帧。
- **范围**: 无
- **作用简述**: 将嘴部区域的切割图像应用到输入图像帧中。

---

### 13. `create_face_mask`
- **函数名**: `create_face_mask`
- **参数**:
  - `face: Face` - 人脸对象。
  - `frame: Frame` - 输入图像帧。
- **返回值**: `np.ndarray` - 返回人脸区域的掩膜。
- **范围**: 无
- **作用简述**: 创建人脸区域的掩膜，用于后续的图像处理。

---

### 14. `apply_color_transfer`
- **函数名**: `apply_color_transfer`
- **参数**:
  - `source: np.ndarray` - 源图像。
  - `target: np.ndarray` - 目标图像。
- **返回值**: `np.ndarray` - 返回颜色转移后的图像。
- **范围**: 无
- **作用简述**: 将目标图像的颜色特征应用到源图像上，实现颜色转移效果。

--- 

以上是所请求的接口文档，涵盖了每个函数的详细信息。