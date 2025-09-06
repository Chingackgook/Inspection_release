# 接口文档

## 1. Encoder 接口

### 1.1 函数：load_model
- **函数名**: `load_model`
- **参数**:
  - `weights_fpath` (Path): 模型权重文件的路径。
  - `device` (Optional): 指定设备，可以是 torch.device 对象或设备名称（如 "cpu", "cuda"）。默认为 None。
- **返回值**: 无
- **范围说明**: 
  - 如果 `device` 为 None，将默认使用可用的 GPU，否则使用 CPU。

#### 调用示例:
```python
load_model(Path("path/to/weights.pth"))
```

### 1.2 函数：is_loaded
- **函数名**: `is_loaded`
- **参数**: 无
- **返回值**: `bool` - 如果模型已加载，返回 True；否则返回 False。
- **范围说明**: 无

#### 调用示例:
```python
if is_loaded():
    print("Model is loaded.")
```

### 1.3 函数：embed_frames_batch
- **函数名**: `embed_frames_batch`
- **参数**:
  - `frames_batch` (numpy.ndarray): 形状为 (batch_size, n_frames, n_channels) 的 mel 频谱图的 numpy 数组。
- **返回值**: `numpy.ndarray` - 形状为 (batch_size, model_embedding_size) 的嵌入特征数组。
- **范围说明**: 
  - 如果模型未加载，将引发异常。

#### 调用示例:
```python
embeddings = embed_frames_batch(mel_spectrograms)
```

### 1.4 函数：compute_partial_slices
- **函数名**: `compute_partial_slices`
- **参数**:
  - `n_samples` (int): 波形中的样本数量。
  - `partial_utterance_n_frames` (Optional, int): 每个部分的 mel 频谱帧数，默认为 `partials_n_frames`。
  - `min_pad_coverage` (Optional, float): 最小填充覆盖率，默认为 0.75。
  - `overlap` (Optional, float): 部分之间的重叠比例，默认为 0.5。
- **返回值**: `Tuple[List[slice], List[slice]]` - 返回波形切片和 mel 频谱切片的列表。
- **范围说明**: 
  - `overlap` 必须在 [0, 1) 范围内。
  - `min_pad_coverage` 必须在 (0, 1] 范围内。

#### 调用示例:
```python
wav_slices, mel_slices = compute_partial_slices(len(wav))
```

### 1.5 函数：embed_utterance
- **函数名**: `embed_utterance`
- **参数**:
  - `wav` (numpy.ndarray): 预处理的波形数组。
  - `using_partials` (Optional, bool): 是否使用部分，默认为 True。
  - `return_partials` (Optional, bool): 是否返回部分嵌入，默认为 False。
  - `**kwargs`: 其他参数传递给 `compute_partial_slices()`。
- **返回值**: `numpy.ndarray` - 形状为 (model_embedding_size,) 的嵌入特征数组。如果 `return_partials` 为 True，还会返回部分嵌入和波形切片。
- **范围说明**: 无

#### 调用示例:
```python
embedding = embed_utterance(wav)
```

### 1.6 函数：plot_embedding_as_heatmap
- **函数名**: `plot_embedding_as_heatmap`
- **参数**:
  - `embed` (numpy.ndarray): 嵌入特征数组。
  - `ax` (Optional): matplotlib 的 Axes 对象，默认为 None。
  - `title` (Optional, str): 图表标题，默认为 ""。
  - `shape` (Optional, Tuple[int, int]): 嵌入的形状，默认为 None。
  - `color_range` (Optional, Tuple[float, float]): 颜色范围，默认为 (0, 0.30)。
- **返回值**: 无
- **范围说明**: 无

#### 调用示例:
```python
plot_embedding_as_heatmap(embedding)
```

### 1.7 函数：preprocess_wav
- **函数名**: `preprocess_wav`
- **参数**:
参数名	类型	是否可选	默认值	说明
fpath_or_wav	Union[str, Path, np.ndarray]	否	-	输入的音频数据，可以是文件路径（支持多种格式）或内存中的音频波形（numpy.ndarray）。
source_sr	Optional[int]	是	None	仅当 fpath_or_wav 是 numpy.ndarray 时有效，表示原始音频的采样率。若为 None 且输入是数组，可能跳过重采样。
normalize	Optional[bool]	是	True	是否对音频进行音量归一化（基于 audio_norm_target_dBFS 目标分贝值）。
trim_silence	Optional[bool]	是	True	是否裁剪长静音段。依赖 webrtcvad 库，若未安装则自动禁用。
返回值​​
类型	说明
np.ndarray	处理后的音频波形，数据类型为 float32，单声道，采样率与全局变量 sampling_rate 一致。

## 2. Synthesizer 接口

### 2.1 初始化方法：__init__
- **函数名**: `__init__`
- **参数**:
  - `model_fpath` (Path): 训练模型文件的路径。
  - `verbose` (Optional, bool): 是否打印详细信息，默认为 True。
- **返回值**: 无
- **范围说明**: 无

#### 调用示例:
```python
synthesizer = Synthesizer(Path("path/to/model.pth"))
```

### 2.2 函数：is_loaded
- **函数名**: `is_loaded`
- **参数**: 无
- **返回值**: `bool` - 如果模型已加载，返回 True；否则返回 False。
- **范围说明**: 无

#### 调用示例:
```python
if synthesizer.is_loaded():
    print("Synthesizer model is loaded.")
```

### 2.3 函数：load
- **函数名**: `load`
- **参数**: 无
- **返回值**: 无
- **范围说明**: 无

#### 调用示例:
```python
synthesizer.load()
```

### 2.4 函数：synthesize_spectrograms
- **函数名**: `synthesize_spectrograms`
- **参数**:
  - `texts` (List[str]): 要合成的文本列表。
  - `embeddings` (Union[np.ndarray, List[np.ndarray]]): 说话人嵌入数组或列表。
  - `return_alignments` (Optional, bool): 是否返回对齐矩阵，默认为 False。
- **返回值**: `List[np.ndarray]` - 合成的 mel 频谱图列表。如果 `return_alignments` 为 True，还会返回对齐矩阵。
- **范围说明**: 无

#### 调用示例:
```python
spectrograms = synthesizer.synthesize_spectrograms(["Hello, world!"], embeddings)
```

### 2.5 函数：load_preprocess_wav
- **函数名**: `load_preprocess_wav`
- **参数**:
  - `fpath` (Union[str, Path]): 音频文件的路径。
- **返回值**: `numpy.ndarray` - 预处理后的波形数组。
- **范围说明**: 无

#### 调用示例:
```python
wav = Synthesizer.load_preprocess_wav("path/to/audio.wav")
```

### 2.6 函数：make_spectrogram
- **函数名**: `make_spectrogram`
- **参数**:
  - `fpath_or_wav` (Union[str, Path, np.ndarray]): 音频文件路径或波形数组。
- **返回值**: `numpy.ndarray` - 生成的 mel 频谱图。
- **范围说明**: 无

#### 调用示例:
```python
mel_spectrogram = Synthesizer.make_spectrogram("path/to/audio.wav")
```

### 2.7 函数：griffin_lim
- **函数名**: `griffin_lim`
- **参数**:
  - `mel` (numpy.ndarray): 输入的 mel 频谱图。
- **返回值**: `numpy.ndarray` - 反转后的波形数组。
- **范围说明**: 无

#### 调用示例:
```python
waveform = Synthesizer.griffin_lim(mel_spectrogram)
```

## 3. Vocoder 接口

### 3.1 函数：load_model
- **函数名**: `load_model`
- **参数**:
  - `weights_fpath` (str): 模型权重文件的路径。
  - `verbose` (Optional, bool): 是否打印详细信息，默认为 True。
- **返回值**: 无
- **范围说明**: 无

#### 调用示例:
```python
load_model("path/to/vocoder_weights.pth")
```

### 3.2 函数：is_loaded
- **函数名**: `is_loaded`
- **参数**: 无
- **返回值**: `bool` - 如果模型已加载，返回 True；否则返回 False。
- **范围说明**: 无

#### 调用示例:
```python
if is_loaded():
    print("Vocoder model is loaded.")
```

### 3.3 函数：infer_waveform
- **函数名**: `infer_waveform`
- **参数**:
  - `mel` (numpy.ndarray): 输入的 mel 频谱图。
  - `normalize` (Optional, bool): 是否归一化，默认为 True。
  - `batched` (Optional, bool): 是否使用批处理，默认为 True。
  - `target` (Optional, int): 目标采样率，默认为 8000。
  - `overlap` (Optional, int): 重叠样本数，默认为 800。
  - `progress_callback` (Optional): 进度回调函数。
- **返回值**: `numpy.ndarray` - 生成的波形数组。
- **范围说明**: 无

#### 调用示例:
```python
waveform = infer_waveform(mel_spectrogram)
```