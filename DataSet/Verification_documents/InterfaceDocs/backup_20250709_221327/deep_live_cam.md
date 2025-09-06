# 接口文档


## 函数：pre_check
### 接口说明
- **函数名**: `pre_check() -> bool`
  - **参数**: 无
  - **返回值**: 返回布尔值，指示预检查是否成功。
  - **范围**: 检查必要的模型文件是否存在。

### 调用示例
```python
if pre_check():
    print("预检查成功")
else:
    print("预检查失败")
```

## 函数：pre_start
### 接口说明
- **函数名**: `pre_start() -> bool`
  - **参数**: 无
  - **返回值**: 返回布尔值，指示预启动检查是否成功。
  - **范围**: 检查源路径和目标路径的有效性。

### 调用示例
```python
if pre_start():
    print("预启动检查成功")
else:
    print("预启动检查失败")
```

## 函数：get_face_swapper
### 接口说明
- **函数名**: `get_face_swapper() -> Any`
  - **参数**: 无
  - **返回值**: 返回面部交换模型，类型为`Any`。
  - **范围**: 获取面部交换模型，确保线程安全。

### 调用示例
```python
face_swapper = get_face_swapper()
```

## 函数：swap_face
### 接口说明
- **函数名**: `swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame`
  - **参数**: 
    - `source_face`: 源面部对象，类型为`Face`。
    - `target_face`: 目标面部对象，类型为`Face`。
    - `temp_frame`: 临时帧，类型为`Frame`（`np.ndarray`）。
  - **返回值**: 返回交换后的帧，类型为`Frame`。
  - **范围**: 进行面部交换操作。

### 调用示例
```python
swapped_frame = swap_face(source_face, target_face, temp_frame)
```

## 函数：process_frame
### 接口说明
- **函数名**: `process_frame(source_face: Face, temp_frame: Frame) -> Frame`
  - **参数**: 
    - `source_face`: 源面部对象，类型为`Face`。
    - `temp_frame`: 临时帧，类型为`Frame`（`np.ndarray`）。
  - **返回值**: 返回处理后的帧，类型为`Frame`。
  - **范围**: 处理单个帧并进行面部交换。

### 调用示例
```python
result_frame = process_frame(source_face, temp_frame)
```

## 函数：process_frame_v2
### 接口说明
- **函数名**: `process_frame_v2(temp_frame: Frame, temp_frame_path: str = "") -> Frame`
  - **参数**: 
    - `temp_frame`: 临时帧，类型为`Frame`（`np.ndarray`）。
    - `temp_frame_path`: 临时帧路径，类型为`str`，默认为空字符串。
  - **返回值**: 返回处理后的帧，类型为`Frame`。
  - **范围**: 处理帧的版本2，支持多种面部映射。

### 调用示例
```python
result_frame_v2 = process_frame_v2(temp_frame, "path/to/temp_frame.jpg")
```

## 函数：process_frames
### 接口说明
- **函数名**: `process_frames(source_path: str, temp_frame_paths: List[str], progress: Any = None) -> None`
  - **参数**: 
    - `source_path`: 源路径，类型为`str`。
    - `temp_frame_paths`: 临时帧路径列表，类型为`List[str]`。
    - `progress`: 进度更新对象，类型为`Any`，默认为`None`。
  - **返回值**: 无返回值。
  - **范围**: 处理多个帧并保存结果。

### 调用示例
```python
process_frames("path/to/source.jpg", ["path/to/frame1.jpg", "path/to/frame2.jpg"])
```

## 函数：process_image
### 接口说明
- **函数名**: `process_image(source_path: str, target_path: str, output_path: str) -> None`
  - **参数**: 
    - `source_path`: 源路径，类型为`str`。
    - `target_path`: 目标路径，类型为`str`。
    - `output_path`: 输出路径，类型为`str`。
  - **返回值**: 无返回值。
  - **范围**: 处理单张图像并保存结果。

### 调用示例
```python
process_image("path/to/source.jpg", "path/to/target.jpg", "path/to/output.jpg")
```

## 函数：process_video
### 接口说明
- **函数名**: `process_video(source_path: str, temp_frame_paths: List[str]) -> None`
  - **参数**: 
    - `source_path`: 源路径，类型为`str`。
    - `temp_frame_paths`: 临时帧路径列表，类型为`List[str]`。
  - **返回值**: 无返回值。
  - **范围**: 处理视频并保存结果。

### 调用示例
```python
process_video("path/to/source_video.mp4", ["path/to/frame1.jpg", "path/to/frame2.jpg"])
```

## 函数：create_lower_mouth_mask
### 接口说明
- **函数名**: `create_lower_mouth_mask(face: Face, frame: Frame) -> (np.ndarray, np.ndarray, tuple, np.ndarray)`
  - **参数**: 
    - `face`: 面部对象，类型为`Face`。
    - `frame`: 帧，类型为`Frame`（`np.ndarray`）。
  - **返回值**: 返回一个元组，包含口罩、口部切割、边界框和下唇多边形。
  - **范围**: 创建下唇区域的掩码。

### 调用示例
```python
mask, mouth_cutout, mouth_box, lower_lip_polygon = create_lower_mouth_mask(face, frame)
```

## 函数：draw_mouth_mask_visualization
### 接口说明
- **函数名**: `draw_mouth_mask_visualization(frame: Frame, face: Face, mouth_mask_data: tuple) -> Frame`
  - **参数**: 
    - `frame`: 帧，类型为`Frame`（`np.ndarray`）。
    - `face`: 面部对象，类型为`Face`。
    - `mouth_mask_data`: 口罩数据元组，类型为`tuple`。
  - **返回值**: 返回可视化后的帧，类型为`Frame`。
  - **范围**: 在帧上绘制下唇掩码的可视化。

### 调用示例
```python
visualized_frame = draw_mouth_mask_visualization(frame, face, mouth_mask_data)
```

## 函数：apply_mouth_area
### 接口说明
- **函数名**: `apply_mouth_area(frame: np.ndarray, mouth_cutout: np.ndarray, mouth_box: tuple, face_mask: np.ndarray, mouth_polygon: np.ndarray) -> np.ndarray`
  - **参数**: 
    - `frame`: 帧，类型为`np.ndarray`。
    - `mouth_cutout`: 口部切割，类型为`np.ndarray`。
    - `mouth_box`: 口部边界框，类型为`tuple`。
    - `face_mask`: 面部掩码，类型为`np.ndarray`。
    - `mouth_polygon`: 口部多边形，类型为`np.ndarray`。
  - **返回值**: 返回应用口部区域后的帧，类型为`np.ndarray`。
  - **范围**: 将口部区域应用于帧。

### 调用示例
```python
final_frame = apply_mouth_area(frame, mouth_cutout, mouth_box, face_mask, mouth_polygon)
```

## 函数：create_face_mask
### 接口说明
- **函数名**: `create_face_mask(face: Face, frame: Frame) -> np.ndarray`
  - **参数**: 
    - `face`: 面部对象，类型为`Face`。
    - `frame`: 帧，类型为`Frame`（`np.ndarray`）。
  - **返回值**: 返回面部掩码，类型为`np.ndarray`。
  - **范围**: 创建面部区域的掩码。

### 调用示例
```python
face_mask = create_face_mask(face, frame)
```

## 函数：apply_color_transfer
### 接口说明
- **函数名**: `apply_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray`
  - **参数**: 
    - `source`: 源图像，类型为`np.ndarray`。
    - `target`: 目标图像，类型为`np.ndarray`。
  - **返回值**: 返回颜色转移后的图像，类型为`np.ndarray`。
  - **范围**: 将目标图像的颜色应用于源图像。

### 调用示例
```python
color_corrected_image = apply_color_transfer(source_image, target_image)
``` 

以上是根据提供的接口实现信息生成的接口文档，涵盖了类、函数的详细说明及调用示例。