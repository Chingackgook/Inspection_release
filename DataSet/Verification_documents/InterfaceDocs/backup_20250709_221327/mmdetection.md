# 接口文档

## DetInferencer 类

### 概述
`DetInferencer` 类是一个用于目标检测推理的接口，能够帮助用户方便地使用预训练模型对输入图像进行目标检测，并支持结果可视化和保存。

### 属性
- `num_visualized_imgs`: 处理的图像数量，用于命名输出图像。
- `num_predicted_imgs`: 预测的图像数量，用于命名输出图像。
- `palette`: 可视化使用的颜色调色板。
- `show_progress`: 控制在推理过程中是否显示进度条。

### 方法

#### `__init__(self, model: Optional[Union[ModelType, str]] = None, weights: Optional[str] = None, device: Optional[str] = None, scope: Optional[str] = 'mmdet', palette: str = 'none', show_progress: bool = True) -> None`
- **参数**:
  - `model`: (str, optional) 配置文件路径或模型名称。
  - `weights`: (str, optional) 检查点路径。
  - `device`: (str, optional) 用于推理的设备。
  - `scope`: (str, optional) 模型的作用域。
  - `palette`: (str) 可视化使用的颜色调色板。
  - `show_progress`: (bool) 控制是否在推理过程中显示进度条。
- **返回值**: None
- **作用**: 初始化 `DetInferencer` 类的实例，设置模型、权重、设备等参数。

#### `_load_weights_to_model(self, model: nn.Module, checkpoint: Optional[dict], cfg: Optional[ConfigType]) -> None`
- **参数**:
  - `model`: (nn.Module) 要加载权重和元信息的模型。
  - `checkpoint`: (dict, optional) 加载的检查点。
  - `cfg`: (Config or ConfigDict, optional) 加载的配置。
- **返回值**: None
- **作用**: 从配置和检查点加载模型权重和元信息。

#### `_init_pipeline(self, cfg: ConfigType) -> Compose`
- **参数**:
  - `cfg`: (ConfigType) 包含测试管道信息的配置。
- **返回值**: Compose
- **作用**: 初始化测试管道。

#### `_get_transform_idx(self, pipeline_cfg: ConfigType, name: Union[str, Tuple[str, type]]) -> int`
- **参数**:
  - `pipeline_cfg`: (ConfigType) 流水线配置。
  - `name`: (Union[str, Tuple[str, type]]) 要查找的变换名称。
- **返回值**: int
- **作用**: 返回流水线中变换的索引，如果未找到则返回 -1。

#### `_init_visualizer(self, cfg: ConfigType) -> Optional[Visualizer]`
- **参数**:
  - `cfg`: (ConfigType) 包含可视化器信息的配置。
- **返回值**: Visualizer or None
- **作用**: 初始化可视化器。

#### `_inputs_to_list(self, inputs: InputsType) -> list`
- **参数**:
  - `inputs`: (InputsType) 用户提供的输入。
- **返回值**: list
- **作用**: 将输入预处理为列表格式。

#### `preprocess(self, inputs: InputsType, batch_size: int = 1, **kwargs)`
- **参数**:
  - `inputs`: (InputsType) 用户提供的输入。
  - `batch_size`: (int) 推理批大小，默认为 1。
- **返回值**: Any
- **作用**: 将输入处理为模型可接受的格式。

#### `_get_chunk_data(self, inputs: Iterable, chunk_size: int)`
- **参数**:
  - `inputs`: (Iterable) 可迭代数据集。
  - `chunk_size`: (int) 批大小。
- **返回值**: list
- **作用**: 从输入中获取批数据。

#### `visualize(self, inputs: InputsType, preds: PredType, return_vis: bool = False, show: bool = False, wait_time: int = 0, draw_pred: bool = True, pred_score_thr: float = 0.3, no_save_vis: bool = False, img_out_dir: str = '', **kwargs) -> Union[List[np.ndarray], None]`
- **参数**:
  - `inputs`: (List[Union[str, np.ndarray]]) 输入数据。
  - `preds`: (List[:obj:`DetDataSample`]) 模型的预测结果。
  - `return_vis`: (bool) 是否返回可视化结果，默认为 False。
  - `show`: (bool) 是否在弹出窗口中显示图像，默认为 False。
  - `wait_time`: (float) 显示间隔时间，默认为 0。
  - `draw_pred`: (bool) 是否绘制预测边界框，默认为 True。
  - `pred_score_thr`: (float) 绘制边界框的最小分数，默认为 0.3。
  - `no_save_vis`: (bool) 是否强制不保存可视化结果，默认为 False。
  - `img_out_dir`: (str) 可视化结果的输出目录，默认为空。
- **返回值**: List[np.ndarray] or None
- **作用**: 可视化预测结果。

#### `postprocess(self, preds: PredType, visualization: Optional[List[np.ndarray]] = None, return_datasamples: bool = False, print_result: bool = False, no_save_pred: bool = False, pred_out_dir: str = '', **kwargs) -> Dict`
- **参数**:
  - `preds`: (List[:obj:`DetDataSample`]) 模型的预测结果。
  - `visualization`: (Optional[np.ndarray]) 可视化结果。
  - `return_datasamples`: (bool) 是否使用 Datasample 存储推理结果，默认为 False。
  - `print_result`: (bool) 是否打印推理结果，默认为 False。
  - `no_save_pred`: (bool) 是否强制不保存预测结果，默认为 False。
  - `pred_out_dir`: (str) 不带可视化的推理结果保存目录，默认为空。
- **返回值**: dict
- **作用**: 处理预测和可视化结果。

#### `pred2dict(self, data_sample: DetDataSample, pred_out_dir: str = '') -> Dict`
- **参数**:
  - `data_sample`: (:obj:`DetDataSample`) 模型的预测结果。
  - `pred_out_dir`: (str) 不带可视化的推理结果保存目录，默认为空。
- **返回值**: dict
- **作用**: 将预测结果提取为字典格式，以便于 JSON 序列化。