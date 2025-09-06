# 接口文档

## 类：OpenVoiceBaseClass

### 初始化方法 `__init__(self, config_path, device='cuda:0')`
- **参数说明**：
  - `config_path` (str): 模型配置文件的路径。
  - `device` (str, optional): 设备类型，默认为 `'cuda:0'`。
- **返回值说明**：无
- **范围说明**：用于初始化基本的语音合成模型及其参数。

### 方法：`load_ckpt(self, ckpt_path)`
- **参数说明**：
  - `ckpt_path` (str): 预训练模型的检查点路径。
- **返回值说明**：无
- **范围说明**：加载预训练模型的状态字典。

---

## 类：BaseSpeakerTTS 继承自 `OpenVoiceBaseClass`

### 初始化方法 `__init__(self, config_path, device='cuda:0')`
- **参数说明**：
  - `config_path` (str): 模型配置文件的路径。
  - `device` (str, optional): 设备类型，默认为 `'cuda:0'`。
- **返回值说明**：无
- **范围说明**：初始化语音合成模型及其参数。

### 方法：`get_text(text, hps, is_symbol)`
- **参数说明**：
  - `text` (str): 输入文本。
  - `hps` (object): 模型超参数对象。
  - `is_symbol` (bool): 指示文本是否为符号的标志。
- **返回值说明**：`torch.LongTensor`: 处理后的文本张量。
- **范围说明**：将文本转换为模型可以理解的格式。

### 方法：`audio_numpy_concat(segment_data_list, sr, speed=1.)`
- **参数说明**：
  - `segment_data_list` (list): 音频片段数组。
  - `sr` (int): 采样率。
  - `speed` (float, optional): 速度因子，默认为 `1.`。
- **返回值说明**：`np.ndarray`: 连接后的音频数据。
- **范围说明**：连接多个音频片段并返回合成的音频numpy数组。

### 方法：`split_sentences_into_pieces(text, language_str)`
- **参数说明**：
  - `text` (str): 输入文本。
  - `language_str` (str): 语言标识符。
- **返回值说明**：`list`: 分割后的句子列表。
- **范围说明**：将输入文本按句子进行分割。

### 方法：`tts(self, text, output_path, speaker, language='English', speed=1.0)`
- **参数说明**：
  - `text` (str): 输入文本。
  - `output_path` (str或None): 输出音频文件的路径。
  - `speaker` (str): 说话者标识符。
  - `language` (str, optional): 语言类型，默认为 `'English'`。
  - `speed` (float, optional): 语速，默认为 `1.0`。
- **返回值说明**：`np.ndarray`或None: 生成的音频数据或无返回值（仅写入文件）。
- **范围说明**：合成文本为音频并保存到指定路径。

---

## 类：ToneColorConverter 继承自 `OpenVoiceBaseClass`

### 初始化方法 `__init__(self, *args, **kwargs)`
- **参数说明**：
  - `*args`: 其他位置参数。
  - `**kwargs`: 其他关键字参数（包括 `enable_watermark`）。
- **返回值说明**：无
- **范围说明**：初始化音调转换模型。

### 方法：`extract_se(self, ref_wav_list, se_save_path=None)`
- **参数说明**：
  - `ref_wav_list` (str或list): 输入参考音频列表，或单个音频文件路径。
  - `se_save_path` (str或None): 保存提取的音频特征的路径，默认为None。
- **返回值说明**：`torch.Tensor`: 提取的音频特征。
- **范围说明**：从参考音频提取音频特征并可保存。

### 方法：`convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default")`
- **参数说明**：
  - `audio_src_path` (str): 源音频文件路径。
  - `src_se` (torch.Tensor): 源音频的特征。
  - `tgt_se` (torch.Tensor): 目标音频的特征。
  - `output_path` (str或None): 输出音频文件的路径，默认为None。
  - `tau` (float, optional): 变换参数，默认为 `0.3`。
  - `message` (str, optional): 水印消息，默认为 "default"。
- **返回值说明**：`np.ndarray`或None: 转换后的音频数据或无返回值（仅写入文件）。
- **范围说明**：将源音频转换为目标风格并可选择添加水印。

### 方法：`add_watermark(self, audio, message)`
- **参数说明**：
  - `audio` (np.ndarray): 输入音频数据。
  - `message` (str): 要添加的水印消息。
- **返回值说明**：`np.ndarray`: 添加水印后的音频数据。
- **范围说明**：将水印信息嵌入到音频数据中。

### 方法：`detect_watermark(self, audio, n_repeat)`
- **参数说明**：
  - `audio` (np.ndarray): 输入音频数据。
  - `n_repeat` (int): 水印信息重复的次数。
- **返回值说明**：`str`: 检测到的水印消息或 "Fail"。
- **范围说明**：从音频中检测水印信息。

---

## 示例调用

```python
# 初始化基类
base = OpenVoiceBaseClass(config_path='path/to/config.yaml')

# 加载检查点
base.load_ckpt(ckpt_path='path/to/checkpoint.pth')

# 初始化说话人TTS
tts = BaseSpeakerTTS(config_path='path/to/config.yaml')

# 生成语音合成
audio_output = tts.tts(text="Hello, world!", output_path='output.wav', speaker='speaker1', language='English', speed=1.0)

# 初始化音调转换器
converter = ToneColorConverter(config_path='path/to/config.yaml')

# 提取音频特征
features = converter.extract_se(ref_wav_list='reference.wav', se_save_path='features.pt')

# 转换音频
converted_audio = converter.convert(audio_src_path='source.wav', src_se=features, tgt_se=some_target_features, output_path='converted.wav', tau=0.3, message='My Watermark')
``` 

以上是接口文档及示例调用，涵盖了所有类及其方法，确保用户能够有效使用。