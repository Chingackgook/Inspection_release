# 接口文档

## 类：IDCreator

### 初始化方法：`__init__`

#### 接口说明
- **函数名**: `__init__`
- **参数**: 无
- **返回值**: 无
- **范围说明**: 初始化IDCreator类的实例，设置回调函数和处理者。

#### 属性说明
- `before_all`: `ContextHandler` - 在所有处理之前的回调函数。
- `after_matting`: `ContextHandler` - 在抠图之后的回调函数。
- `after_detect`: `ContextHandler` - 在人脸检测之后的回调函数。
- `after_all`: `ContextHandler` - 在所有处理之后的回调函数。
- `matting_handler`: `ContextHandler` - 处理人像抠图的处理者，默认为`extract_human`。
- `detection_handler`: `ContextHandler` - 处理人脸检测的处理者，默认为`detect_face_mtcnn`。
- `beauty_handler`: `ContextHandler` - 处理美颜的处理者，默认为`beauty_face`。
- `ctx`: `Context` - 存储处理上下文的对象。

### 方法：`__call__`

#### 接口说明
- **函数名**: `__call__`
- **参数**:
  - `image` (np.ndarray): 输入图像。
  - `size` (Tuple[int, int]): 输出的图像大小，默认为(413, 295)。
  - `change_bg_only` (bool): 是否只需要抠图，默认为False。
  - `crop_only` (bool): 是否只需要裁剪，默认为False。
  - `head_measure_ratio` (float): 人脸面积与全图面积的期望比值，默认为0.2。
  - `head_height_ratio` (float): 人脸中心处在全图高度的比例期望值，默认为0.45。
  - `head_top_range` (Tuple[float, float]): 头距离顶部的比例（max,min），默认为(0.12, 0.1)。
  - `face` (Tuple[int, int, int, int]): 人脸坐标，默认为None。
  - `whitening_strength` (int): 美白强度，默认为0。
  - `brightness_strength` (int): 亮度强度，默认为0。
  - `contrast_strength` (int): 对比度强度，默认为0。
  - `sharpen_strength` (int): 锐化强度，默认为0。
  - `saturation_strength` (int): 饱和度强度，默认为0。
  - `face_alignment` (bool): 是否需要人脸矫正，默认为False。
- **返回值**: `Result` - 返回处理后的证件照和一系列参数。
- **范围说明**: 处理输入图像，生成证件照，支持多种参数配置。

## 函数：示例调用

### 示例代码

```python
from hivision import IDCreator
creator = IDCreator()
choose_handler(creator, args.matting_model, args.face_detect_model)

root_dir = os.path.dirname(os.path.abspath(__file__))
input_image = cv2.imread(args.input_image_dir, cv2.IMREAD_UNCHANGED)

# 如果模式是生成证件照
if args.type == "idphoto":
    # 将字符串转为元组
    size = (int(args.height), int(args.width))
    try:
        result = creator(input_image, size=size, face_alignment=args.face_align)
    except FaceError:
        print("人脸数量不等于 1，请上传单张人脸的图像。")
    else:
        # 保存标准照
        save_image_dpi_to_bytes(cv2.cvtColor(result.standard, cv2.COLOR_RGBA2BGRA), args.output_image_dir, dpi=args.dpi)

        # 保存高清照
        file_name, file_extension = os.path.splitext(args.output_image_dir)
        new_file_name = file_name + "_hd" + file_extension
        save_image_dpi_to_bytes(cv2.cvtColor(result.hd, cv2.COLOR_RGBA2BGRA), new_file_name, dpi=args.dpi)
```

### 说明
以上示例展示了如何使用`IDCreator`类生成证件照。首先创建`IDCreator`的实例，然后选择处理模型，读取输入图像，调用`__call__`方法生成证件照，并处理可能的异常。最后，保存生成的标准照和高清照。