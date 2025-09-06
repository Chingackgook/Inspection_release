# 接口文档

## 类：`MusicGen`

### 初始化方法：`__init__`
- **函数名**: `__init__`
- **参数说明**:
  - `name` (str): 模型名称。
  - `compression_model` (CompressionModel): 用于将音频映射到可逆离散表示的压缩模型。
  - `lm` (LMModel): 在离散表示上进行语言建模的模型。
  - `max_duration` (float, optional): 模型可以生成的最大时长，默认为训练参数推断的值。
- **返回值说明**: 无
- **范围说明**: 初始化`MusicGen`类的实例。

### 方法：`get_pretrained`
- **函数名**: `get_pretrained`
- **参数说明**:
  - `name` (str, optional): 预训练模型的名称，默认为 `'facebook/musicgen-melody'`。
  - `device` (optional): 设备类型，默认为 `None`，会根据可用的CUDA设备自动选择。
- **返回值说明**: 返回一个预训练的`MusicGen`模型实例。
- **范围说明**: 获取预训练模型。

### 方法：`set_generation_params`
- **函数名**: `set_generation_params`
- **参数说明**:
  - `use_sampling` (bool, optional): 是否使用采样，默认为 `True`。
  - `top_k` (int, optional): 采样时使用的top_k值，默认为 `250`。
  - `top_p` (float, optional): 采样时使用的top_p值，默认为 `0.0`。
  - `temperature` (float, optional): Softmax温度参数，默认为 `1.0`。
  - `duration` (float, optional): 生成波形的时长，默认为 `30.0`。
  - `cfg_coef` (float, optional): 用于无分类器引导的系数，默认为 `3.0`。
  - `cfg_coef_beta` (float, optional): 双分类器引导中的beta系数，默认为 `None`。
  - `two_step_cfg` (bool, optional): 是否执行双步分类器引导，默认为 `False`。
  - `extend_stride` (float, optional): 扩展生成时的步幅，默认为 `18`。
- **返回值说明**: 无
- **范围说明**: 设置生成参数。

### 方法：`set_style_conditioner_params`
- **函数名**: `set_style_conditioner_params`
- **参数说明**:
  - `eval_q` (int): 用于量化风格条件的残差量化流的数量，默认为 `3`。
  - `excerpt_length` (float): 从音频中提取的摘录长度（秒），默认为 `3.0`。
  - `ds_factor` (int, optional): 用于下采样风格标记的因子，默认为 `None`。
  - `encodec_n_q` (int, optional): 如果使用encodec作为特征提取器，设置用于提取特征的流的数量，默认为 `None`。
- **返回值说明**: 无
- **范围说明**: 设置风格条件器的参数。

### 方法：`generate_with_chroma`
- **函数名**: `generate_with_chroma`
- **参数说明**:
  - `descriptions` (list of str): 用作文本条件的字符串列表。
  - `melody_wavs` (MelodyType): 用作旋律条件的波形批次。
  - `melody_sample_rate` (int): 旋律波形的采样率。
  - `progress` (bool, optional): 是否显示生成过程的进度，默认为 `False`。
  - `return_tokens` (bool, optional): 是否返回生成的tokens，默认为 `False`。
- **返回值说明**: 返回生成的音频波形，或生成的音频波形和tokens的元组（如果`return_tokens`为`True`）。
- **范围说明**: 根据文本和旋律生成样本。

### 方法：`_prepare_tokens_and_attributes`
- **函数名**: `_prepare_tokens_and_attributes`
- **参数说明**:
  - `descriptions` (list of str): 用作文本条件的字符串列表。
  - `prompt` (torch.Tensor): 用于续写的波形批次。
  - `melody_wavs` (MelodyList, optional): 用作旋律条件的波形批次，默认为 `None`。
- **返回值说明**: 返回准备好的属性和可选的提示tokens。
- **范围说明**: 准备模型输入。

### 方法：`_generate_tokens`
- **函数名**: `_generate_tokens`
- **参数说明**:
  - `attributes` (list of ConditioningAttributes): 用于生成的条件（文本/旋律）。
  - `prompt_tokens` (torch.Tensor, optional): 用于续写的音频提示，默认为 `None`。
  - `progress` (bool, optional): 是否显示生成过程的进度，默认为 `False`。
- **返回值说明**: 返回生成的音频tokens，形状为 `[B, C, T]`。
- **范围说明**: 生成离散音频tokens。

## 调用示例

```python
# 导入必要的库
import torch
from your_module import MusicGen

# 获取预训练模型
model = MusicGen.get_pretrained(name='facebook/musicgen-medium')

# 设置生成参数
model.set_generation_params(duration=30.0, temperature=0.8)

# 生成音乐
descriptions = ["A calm and soothing melody", "An upbeat and lively tune"]
melody_wavs = [torch.randn(1, 2, 44100), None]  # 示例旋律波形
generated_audio = model.generate_with_chroma(descriptions, melody_wavs, melody_sample_rate=44100)

# 处理生成的音频（例如保存或播放）
```