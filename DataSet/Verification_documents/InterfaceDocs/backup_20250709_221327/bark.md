# 接口文档

## 1. 接口：`generate_audio`

### 函数说明
- **函数名**: `generate_audio`
- **参数说明**:
  - `text` (str): 要转换为音频的文本。
  - `history_prompt` (Optional[Union[Dict, str]]): 用于音频克隆的历史选择。
  - `text_temp` (float): 生成温度（1.0 更具多样性，0.0 更保守）。
  - `waveform_temp` (float): 生成温度（1.0 更具多样性，0.0 更保守）。
  - `silent` (bool): 禁用进度条。
  - `output_full` (bool): 返回完整生成以用作历史提示。
- **返回值说明**:
  - 返回一个 numpy 音频数组，采样频率为 24kHz。
- **范围说明**: 
  - 适用于音频生成任务。

### 调用示例
```python
audio_array = generate_audio("Hello, world!", text_temp=0.7, waveform_temp=0.7)
```

---

## 2. 接口：`text_to_semantic`

### 函数说明
- **函数名**: `text_to_semantic`
- **参数说明**:
  - `text` (str): 要转换为语义的文本。
  - `history_prompt` (Optional[Union[Dict, str]]): 用于音频克隆的历史选择。
  - `temp` (float): 生成温度（1.0 更具多样性，0.0 更保守）。
  - `silent` (bool): 禁用进度条。
- **返回值说明**:
  - 返回一个 numpy 语义数组，用于输入到 `semantic_to_waveform`。
- **范围说明**: 
  - 适用于文本到语义的转换。

### 调用示例
```python
semantic_array = text_to_semantic("Hello, world!")
```

---

## 3. 接口：`semantic_to_waveform`

### 函数说明
- **函数名**: `semantic_to_waveform`
- **参数说明**:
  - `semantic_tokens` (np.ndarray): 从 `text_to_semantic` 输出的语义令牌。
  - `history_prompt` (Optional[Union[Dict, str]]): 用于音频克隆的历史选择。
  - `temp` (float): 生成温度（1.0 更具多样性，0.0 更保守）。
  - `silent` (bool): 禁用进度条。
  - `output_full` (bool): 返回完整生成以用作历史提示。
- **返回值说明**:
  - 返回一个 numpy 音频数组，采样频率为 24kHz。
- **范围说明**: 
  - 适用于语义到音频的转换。

### 调用示例
```python
audio_array = semantic_to_waveform(semantic_array)
```

---

## 4. 接口：`save_as_prompt`

### 函数说明
- **函数名**: `save_as_prompt`
- **参数说明**:
  - `filepath` (str): 要保存的文件路径，必须以 ".npz" 结尾。
  - `full_generation` (dict): 包含生成的完整信息的字典。
- **返回值说明**:
  - 无返回值。
- **范围说明**: 
  - 适用于将生成的完整信息保存为文件。

### 调用示例
```python
save_as_prompt("output.npz", full_generation)
```