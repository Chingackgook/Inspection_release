```markdown
# 接口文档

## SAM 类

### 概述
SAM (Segment Anything Model) 是一个用于可提示的实时图像分割的接口类。它支持多种提示方式，如边界框、点或标签，并具备零-shot 性能，经过 SA-1B 数据集训练。

### 初始化方法

#### `__init__(self, model="sam_b.pt")`
- **参数**:
  - `model` (str): 预训练 SAM 模型文件的路径。文件应具有 .pt 或 .pth 扩展名。
- **返回值**: None
- **作用**: 初始化 SAM 模型，加载指定的预训练模型。

### 属性

#### `names`
- **返回值**: list | None
- **作用**: 获取与加载模型相关的类名。如果模型中定义了类名，则返回类名；否则返回 None。

#### `device`
- **返回值**: torch.device | None
- **作用**: 获取模型参数所在的设备（CPU/GPU）。仅适用于 nn.Module 实例的模型。

#### `transforms`
- **返回值**: object | None
- **作用**: 获取加载模型的输入数据所应用的变换。如果模型中定义了变换，则返回变换对象；否则返回 None。

#### `task_map`
- **返回值**: dict
- **作用**: 提供从 'segment' 任务到其对应的 'Predictor' 的映射。

### 公有方法

#### `predict(self, source, stream=False, bboxes=None, points=None, labels=None, **kwargs)`
- **参数**:
  - `source` (str): 图像或视频文件的路径，或 PIL.Image 对象，或 numpy.ndarray 对象。
  - `stream` (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
  - `bboxes` (list, optional): 用于提示分割的边界框坐标列表。默认为 None。
  - `points` (list, optional): 用于提示分割的点列表。默认为 None。
  - `labels` (list, optional): 用于提示分割的标签列表。默认为 None。
- **返回值**: list
- **作用**: 对给定的图像或视频源执行分割预测。

#### `__call__(self, source=None, stream=False, bboxes=None, points=None, labels=None, **kwargs)`
- **参数**:
  - `source` (str): 图像或视频文件的路径，或 PIL.Image 对象，或 numpy.ndarray 对象。
  - `stream` (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
  - `bboxes` (list, optional): 用于提示分割的边界框坐标列表。默认为 None。
  - `points` (list, optional): 用于提示分割的点列表。默认为 None。
  - `labels` (list, optional): 用于提示分割的标签列表。默认为 None。
- **返回值**: list
- **作用**: `predict` 方法的别名，执行分割预测。

#### `info(self, detailed=False, verbose=True)`
- **参数**:
  - `detailed` (bool, optional): 如果为 True，则显示模型的详细信息。默认为 False。
  - `verbose` (bool, optional): 如果为 True，则在控制台显示信息。默认为 True。
- **返回值**: tuple
- **作用**: 记录有关 SAM 模型的信息。

### 继承自 Model 类的方法

以下方法继承自 `Model` 类，并在 `SAM` 类中可用：

#### `reset_weights(self) -> "Model"`
- **返回值**: Model
- **作用**: 重置模型的权重。

#### `load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model"`
- **参数**:
  - `weights` (Union[str, Path]): 权重文件的路径。默认为 "yolov8n.pt"。
- **返回值**: Model
- **作用**: 加载指定的权重到模型中。

#### `save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None`
- **参数**:
  - `filename` (Union[str, Path]): 保存的文件名。默认为 "saved_model.pt"。
  - `use_dill` (bool): 是否使用 dill 序列化。默认为 True。
- **返回值**: None
- **作用**: 保存模型的当前状态到指定文件。

#### `fuse(self)`
- **返回值**: None
- **作用**: 融合模型的层以提高推理速度。

#### `embed(self, source, stream=False, **kwargs) -> list`
- **参数**:
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]): 输入源。
  - `stream` (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
- **返回值**: list
- **作用**: 嵌入模型的输出。

#### `track(self, source=None, stream=False, persist=False, **kwargs) -> list`
- **参数**:
  - `source` (Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor]): 输入源。
  - `stream` (bool, optional): 如果为 True，则启用实时流处理。默认为 False。
  - `persist` (bool, optional): 如果为 True，则保持跟踪状态。默认为 False。
- **返回值**: list
- **作用**: 执行目标跟踪。

#### `val(self, validator=None, **kwargs)`
- **参数**:
  - `validator` (optional): 验证器对象。
- **返回值**: None
- **作用**: 执行模型验证。

#### `benchmark(self, **kwargs)`
- **参数**:
  - `kwargs`: 其他参数。
- **返回值**: None
- **作用**: 执行模型基准测试。

#### `export(self, **kwargs)`
- **参数**:
  - `kwargs`: 其他参数。
- **返回值**: None
- **作用**: 导出模型。

#### `train(self, trainer=None, **kwargs)`
- **参数**:
  - `trainer` (optional): 训练器对象。
- **返回值**: None
- **作用**: 执行模型训练。

#### `tune(self, use_ray=False, iterations=10, *args, **kwargs)`
- **参数**:
  - `use_ray` (bool, optional): 是否使用 Ray 进行调优。默认为 False。
  - `iterations` (int, optional): 调优的迭代次数。默认为 10。
- **返回值**: None
- **作用**: 执行模型调优。

#### `add_callback(self, event: str, func) -> None`
- **参数**:
  - `event` (str): 要附加回调的事件名称。
  - `func` (callable): 要注册的回调函数。
- **返回值**: None
- **作用**: 为指定事件添加回调函数。

#### `clear_callback(self, event: str) -> None`
- **参数**:
  - `event` (str): 要清除回调的事件名称。
- **返回值**: None
- **作用**: 清除指定事件的所有回调函数。

#### `reset_callbacks(self) -> None`
- **返回值**: None
- **作用**: 重置所有回调为默认函数。

```