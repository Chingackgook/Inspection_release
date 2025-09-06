# 接口文档

## 函数: `get_watermarked`

### 参数
- `pil_image` (Image): 要添加水印的 PIL 图像。

### 返回值
- (Image): 返回添加水印后的图像。如果发生错误，则返回原始图像。

### 范围
- 该函数使用 OpenCV 库处理图像，添加水印并返回处理后的图像。

---

## 类: `ModelImageVisualizer`

### 初始化方法: `__init__`

#### 参数
- `filter` (IFilter): 用于图像处理的滤镜实例。
- `results_dir` (str, 可选): 结果保存的目录路径。如果为 None，则使用默认路径。

#### 属性
- `filter`: 存储传入的滤镜实例。
- `results_dir`: 存储结果保存的目录路径。

### 公有方法

#### 方法: `plot_transformed_image_from_url`

##### 参数
- `url` (str): 图像的 URL 地址。
- `path` (str, 可选): 保存图像的路径，默认为 'test_images/image.png'。
- `results_dir` (Path, 可选): 结果保存的目录路径。
- `figsize` (Tuple[int, int], 可选): 图像显示的大小，默认为 (20, 20)。
- `render_factor` (int, 可选): 渲染因子，用于图像处理。
- `display_render_factor` (bool, 可选): 是否显示渲染因子，默认为 False。
- `compare` (bool, 可选): 是否进行比较显示，默认为 False。
- `post_process` (bool, 可选): 是否进行后处理，默认为 True。
- `watermarked` (bool, 可选): 是否添加水印，默认为 True。

##### 返回值
- (Path): 返回处理后图像的保存路径。

#### 方法: `plot_transformed_image`

##### 参数
- `path` (str): 要处理的图像路径。
- `results_dir` (Path, 可选): 结果保存的目录路径。
- `figsize` (Tuple[int, int], 可选): 图像显示的大小，默认为 (20, 20)。
- `render_factor` (int, 可选): 渲染因子，用于图像处理。
- `display_render_factor` (bool, 可选): 是否显示渲染因子，默认为 False。
- `compare` (bool, 可选): 是否进行比较显示，默认为 False。
- `post_process` (bool, 可选): 是否进行后处理，默认为 True。
- `watermarked` (bool, 可选): 是否添加水印，默认为 True。

##### 返回值
- (Path): 返回处理后图像的保存路径。

#### 方法: `_plot_comparison`

##### 参数
- `figsize` (Tuple[int, int]): 图像显示的大小。
- `render_factor` (int): 渲染因子。
- `display_render_factor` (bool): 是否显示渲染因子。
- `orig` (Image): 原始图像。
- `result` (Image): 处理后的图像。

##### 返回值
- None

#### 方法: `_plot_solo`

##### 参数
- `figsize` (Tuple[int, int]): 图像显示的大小。
- `render_factor` (int): 渲染因子。
- `display_render_factor` (bool): 是否显示渲染因子。
- `result` (Image): 处理后的图像。

##### 返回值
- None

#### 方法: `_save_result_image`

##### 参数
- `source_path` (Path): 原始图像的路径。
- `image` (Image): 处理后的图像。
- `results_dir` (Path, 可选): 结果保存的目录路径。

##### 返回值
- (Path): 返回处理后图像的保存路径。

#### 方法: `get_transformed_image`

##### 参数
- `path` (Path): 要处理的图像路径。
- `render_factor` (int, 可选): 渲染因子，用于图像处理。
- `post_process` (bool, 可选): 是否进行后处理，默认为 True。
- `watermarked` (bool, 可选): 是否添加水印，默认为 True。

##### 返回值
- (Image): 返回处理后的图像。

#### 方法: `_plot_image`

##### 参数
- `image` (Image): 要显示的图像。
- `render_factor` (int): 渲染因子。
- `axes` (Axes, 可选): Matplotlib 轴对象。
- `figsize` (Tuple[int, int], 可选): 图像显示的大小，默认为 (20, 20)。
- `display_render_factor` (bool): 是否显示渲染因子。

##### 返回值
- None

#### 方法: `_get_num_rows_columns`

##### 参数
- `num_images` (int): 图像数量。
- `max_columns` (int): 最大列数。

##### 返回值
- (Tuple[int, int]): 返回行数和列数的元组。