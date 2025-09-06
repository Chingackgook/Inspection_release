# 接口文档

## 类：`Separator`

### 初始化方法：`__init__`

#### 参数说明：
- `params_descriptor` (str): 
  - 描述用于构建模型的 TensorFlow 参数的描述符。
  
- `MWF` (bool, 可选): 
  - 指定是否启用 Wiener 过滤器，默认为 `False`。
  
- `multiprocess` (bool, 可选): 
  - 指定是否启用多进程，默认为 `True`。

#### 属性：
- `_params` (Dict): 
  - 存储加载的模型参数。
  
- `_sample_rate` (int): 
  - 音频采样率。
  
- `_MWF` (bool): 
  - 是否启用 Wiener 过滤器。
  
- `_tf_graph` (tf.Graph): 
  - TensorFlow 图对象。
  
- `_prediction_generator` (Optional[Generator]): 
  - 用于生成预测的生成器。
  
- `_input_provider` (Optional): 
  - 输入提供者。
  
- `_builder` (Optional): 
  - 估计器规格构建器。
  
- `_features` (Optional): 
  - 特征占位符。
  
- `_session` (Optional[tf.Session]): 
  - TensorFlow 会话对象。
  
- `_pool` (Optional[Pool]): 
  - 进程池对象。
  
- `_tasks` (List): 
  - 存储任务的列表。
  
- `estimator` (Optional): 
  - TensorFlow 估计器对象。

### 方法：`_get_prediction_generator`

#### 参数说明：
- `data` (dict): 
  - 包含输入数据的字典。

#### 返回值说明：
- `Generator`: 
  - 返回一个生成器，用于生成预测结果。

### 方法：`join`

#### 参数说明：
- `timeout` (int, 可选): 
  - 等待任务完成的超时时间，默认为 200 秒。

#### 返回值说明：
- `None`: 
  - 此方法没有返回值。

### 方法：`_get_input_provider`

#### 返回值说明：
- `InputProvider`: 
  - 返回输入提供者实例。

### 方法：`_get_features`

#### 返回值说明：
- `Dict`: 
  - 返回特征占位符的字典。

### 方法：`_get_builder`

#### 返回值说明：
- `EstimatorSpecBuilder`: 
  - 返回估计器规格构建器实例。

### 方法：`_get_session`

#### 返回值说明：
- `tf.Session`: 
  - 返回 TensorFlow 会话对象。

### 方法：`_separate_tensorflow`

#### 参数说明：
- `waveform` (np.ndarray): 
  - 要分离的波形（numpy 数组）。
  
- `audio_descriptor` (AudioDescriptor): 
  - 用于分离的音频描述符。

#### 返回值说明：
- `Dict`: 
  - 返回分离后的波形字典。

### 方法：`separate`

#### 参数说明：
- `waveform` (np.ndarray): 
  - 要分离的波形（numpy 数组）。
  
- `audio_descriptor` (Optional[str]): 
  - 描述波形的字符串（例如文件名），可选。

#### 返回值说明：
- `Dict`: 
  - 返回分离后的波形字典。

### 方法：`separate_to_file`

#### 参数说明：
- `audio_descriptor` (AudioDescriptor): 
  - 描述要分离的音频，供音频适配器使用。
  
- `destination` (str): 
  - 输出目标目录。
  
- `audio_adapter` (Optional[AudioAdapter]): 
  - 可选的音频适配器，用于 I/O。
  
- `offset` (float, 可选): 
  - 加载音频的偏移量，默认为 0。
  
- `duration` (float, 可选): 
  - 加载音频的持续时间，默认为 600 秒。
  
- `codec` (Codec, 可选): 
  - 导出音频的编码格式。
  
- `bitrate` (str, 可选): 
  - 导出音频的比特率，默认为 "128k"。
  
- `filename_format` (str, 可选): 
  - 文件名格式，默认为 "{filename}/{instrument}.{codec}"。
  
- `synchronous` (bool, 可选): 
  - 是否同步执行，默认为 `True`。

#### 返回值说明：
- `None`: 
  - 此方法没有返回值。

### 方法：`save_to_file`

#### 参数说明：
- `sources` (Dict): 
  - 要导出的源字典，键为乐器名称，值为对应的波形（N x 2 numpy 数组）。
  
- `audio_descriptor` (AudioDescriptor): 
  - 描述要分离的音频，供音频适配器使用。
  
- `destination` (str): 
  - 输出目标目录。
  
- `filename_format` (str, 可选): 
  - 文件名格式，默认为 "{filename}/{instrument}.{codec}"。
  
- `codec` (Codec, 可选): 
  - 导出音频的编码格式。
  
- `audio_adapter` (Optional[AudioAdapter]): 
  - 可选的音频适配器，用于 I/O。
  
- `bitrate` (str, 可选): 
  - 导出音频的比特率，默认为 "128k"。
  
- `synchronous` (bool, 可选): 
  - 是否同步执行，默认为 `True`。

#### 返回值说明：
- `None`: 
  - 此方法没有返回值。

## 函数：`create_estimator`

#### 参数说明：
- `params` (Dict): 
  - 用于构建模型的参数字典。
  
- `MWF` (bool): 
  - 指定是否启用 Wiener 过滤器。

#### 返回值说明：
- `tf.Tensor`: 
  - 返回一个 TensorFlow 估计器对象。