# 接口文档

## 1. 类：Whisper

### 1.1 初始化方法

#### 方法名
`__init__`

#### 参数说明
- `dims: ModelDimensions` - 模型的维度信息。

#### 返回值说明
无返回值。

---

### 1.2 方法：load_state_dict

#### 方法名
`load_state_dict`

#### 参数说明
- `state_dict: dict` - 包含模型权重的状态字典。

#### 返回值说明
无返回值。

---

### 1.3 方法：set_alignment_heads

#### 方法名
`set_alignment_heads`

#### 参数说明
- `alignment_heads: bytes` - 表示交叉注意力头的布尔数组，指示与单词级时间对齐高度相关的头。

#### 返回值说明
无返回值。

---

### 1.4 方法：detect_language

#### 方法名
`detect_language`

#### 参数说明
- `mel_segment: torch.Tensor` - 输入的梅尔频谱段。

#### 返回值说明
- `Tuple[str, dict]` - 检测到的语言及其概率字典。

---

### 1.5 方法：decode

#### 方法名
`decode`

#### 参数说明
- `segment: torch.Tensor` - 输入的梅尔频谱段。
- `options: DecodingOptions` - 解码选项。

#### 返回值说明
- `DecodingResult` - 解码结果对象。

---

## 2. 类：ModelDimensions

### 2.1 初始化方法

#### 方法名
`__init__`

#### 参数说明
- `n_mels: int` - 梅尔频谱的维度。
- `n_audio_ctx: int` - 音频上下文的维度。
- `n_text_ctx: int` - 文本上下文的维度。
- `num_languages: int` - 支持的语言数量。

#### 返回值说明
无返回值。

---

## 3. 类：DecodingOptions

### 3.1 初始化方法

#### 方法名
`__init__`

#### 参数说明
- `temperature: float` - 采样温度。
- `language: str` - 语言选项。
- `task: str` - 任务类型。
- `fp16: bool` - 是否使用FP16。
- 其他关键字参数。

#### 返回值说明
无返回值。

---

## 4. 类：DecodingResult

### 4.1 初始化方法

#### 方法名
`__init__`

#### 参数说明
- `tokens: List[int]` - 解码得到的token列表。
- `temperature: float` - 使用的温度。
- `avg_logprob: float` - 平均对数概率。
- `compression_ratio: float` - 压缩比。
- `no_speech_prob: float` - 无语音概率。

#### 返回值说明
无返回值。

---

## 5. 函数：load_model

### 5.1 方法名
`load_model`

### 5.2 参数说明
- `name: str` - 模型名称或路径。
- `device: Optional[Union[str, torch.device]]` - PyTorch设备。
- `download_root: str` - 下载模型文件的路径。
- `in_memory: bool` - 是否将模型权重预加载到内存中。

### 5.3 返回值说明
- `Whisper` - Whisper ASR模型实例。

### 5.4 调用示例
```python
model = load_model("medium")
```

---

## 6. 函数：transcribe

### 6.1 方法名
`transcribe`

### 6.2 参数说明
- `model: Whisper` - Whisper模型实例。
- `audio: Union[str, np.ndarray, torch.Tensor]` - 音频文件路径或音频波形。
- `verbose: Optional[bool]` - 是否显示解码文本。
- `temperature: Union[float, Tuple[float, ...]]` - 采样温度。
- `compression_ratio_threshold: Optional[float]` - 压缩比阈值。
- `logprob_threshold: Optional[float]` - 对数概率阈值。
- `no_speech_threshold: Optional[float]` - 无语音阈值。
- `condition_on_previous_text: bool` - 是否基于前文进行解码。
- `initial_prompt: Optional[str]` - 初始提示文本。
- `word_timestamps: bool` - 是否提取单词级时间戳。
- `prepend_punctuations: str` - 前置标点符号。
- `append_punctuations: str` - 后置标点符号。
- `clip_timestamps: Union[str, List[float]]` - 处理的片段时间戳。
- `hallucination_silence_threshold: Optional[float]` - 幻觉静音阈值。
- `**decode_options` - 解码选项的其他关键字参数。

### 6.3 返回值说明
- `dict` - 包含结果文本、段落级细节和检测到的语言。

### 6.4 调用示例
```python
result = transcribe(model, "audio_file.wav", verbose=True)
```