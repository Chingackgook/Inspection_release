# CosyVoice2 API 接口文档

## 类：CosyVoice

### 初始化方法：`__init__(model_dir, load_jit=False, load_trt=False, fp16=False)`

- **参数说明**：
  - `model_dir` (str): 模型目录的路径。
  - `load_jit` (bool): 是否加载 JIT 模型，默认为 False。
  - `load_trt` (bool): 是否加载 TRT 模型，默认为 False。
  - `fp16` (bool): 是否使用 FP16 精度，默认为 False。

- **返回值说明**：无返回值。

### 属性

- `instruct` (bool): 指示模型是否支持指令推理。
- `model_dir` (str): 模型目录的路径。
- `fp16` (bool): 是否使用 FP16 精度。
- `frontend` (CosyVoiceFrontEnd): 前端处理模块。
- `sample_rate` (int): 采样率。
- `model` (CosyVoiceModel): 语音合成模型。

### 方法

#### 方法：`list_available_spks()`

- **参数说明**：无参数。
- **返回值说明**：返回可用的说话人列表 (list)。

#### 方法：`add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)`

- **参数说明**：
  - `prompt_text` (str): 提示文本。
  - `prompt_speech_16k` (Tensor): 提示语音，采样率为 16kHz。
  - `zero_shot_spk_id` (str): 零样本说话人 ID。

- **返回值说明**：返回布尔值，表示操作是否成功。

#### 方法：`save_spkinfo()`

- **参数说明**：无参数。
- **返回值说明**：无返回值。

#### 方法：`inference_sft(tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)`

- **参数说明**：
  - `tts_text` (str): 要合成的文本。
  - `spk_id` (str): 说话人 ID。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。
  - `text_frontend` (bool): 是否使用文本前端处理，默认为 True。

- **返回值说明**：返回生成的语音输出 (Generator)。

#### 方法：`inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`

- **参数说明**：
  - `tts_text` (str): 要合成的文本。
  - `prompt_text` (str): 提示文本。
  - `prompt_speech_16k` (Tensor): 提示语音，采样率为 16kHz。
  - `zero_shot_spk_id` (str): 零样本说话人 ID，默认为空。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。
  - `text_frontend` (bool): 是否使用文本前端处理，默认为 True。

- **返回值说明**：返回生成的语音输出 (Generator)。

#### 方法：`inference_cross_lingual(tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`

- **参数说明**：
  - `tts_text` (str): 要合成的文本。
  - `prompt_speech_16k` (Tensor): 提示语音，采样率为 16kHz。
  - `zero_shot_spk_id` (str): 零样本说话人 ID，默认为空。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。
  - `text_frontend` (bool): 是否使用文本前端处理，默认为 True。

- **返回值说明**：返回生成的语音输出 (Generator)。

#### 方法：`inference_instruct(tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True)`

- **参数说明**：
  - `tts_text` (str): 要合成的文本。
  - `spk_id` (str): 说话人 ID。
  - `instruct_text` (str): 指令文本。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。
  - `text_frontend` (bool): 是否使用文本前端处理，默认为 True。

- **返回值说明**：返回生成的语音输出 (Generator)。

#### 方法：`inference_vc(source_speech_16k, prompt_speech_16k, stream=False, speed=1.0)`

- **参数说明**：
  - `source_speech_16k` (Tensor): 源语音，采样率为 16kHz。
  - `prompt_speech_16k` (Tensor): 提示语音，采样率为 16kHz。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。

- **返回值说明**：返回生成的语音输出 (Generator)。

---

## 类：CosyVoice2

### 初始化方法：`__init__(model_dir, load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)`

- **参数说明**：
  - `model_dir` (str): 模型目录的路径。
  - `load_jit` (bool): 是否加载 JIT 模型，默认为 False。
  - `load_trt` (bool): 是否加载 TRT 模型，默认为 False。
  - `fp16` (bool): 是否使用 FP16 精度，默认为 False。
  - `use_flow_cache` (bool): 是否使用流缓存，默认为 False。

- **返回值说明**：无返回值。

### 方法

#### 方法：`inference_instruct(self, *args, **kwargs)`

- **参数说明**：无参数。
- **返回值说明**：抛出 NotImplementedError，表示该方法未实现。

#### 方法：`inference_instruct2(tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`

- **参数说明**：
  - `tts_text` (str): 要合成的文本。
  - `instruct_text` (str): 指令文本。
  - `prompt_speech_16k` (Tensor): 提示语音，采样率为 16kHz。
  - `zero_shot_spk_id` (str): 零样本说话人 ID，默认为空。
  - `stream` (bool): 是否流式输出，默认为 False。
  - `speed` (float): 合成速度，默认为 1.0。
  - `text_frontend` (bool): 是否使用文本前端处理，默认为 True。

- **返回值说明**：返回生成的语音输出 (Generator)。

---

## 示例代码

```python
from modelscope import snapshot_download
import sys
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio

# 初始化 CosyVoice2
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, use_flow_cache=False)

# 零样本使用示例
prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

# 保存零样本说话人以备后用
assert cosyvoice.add_zero_shot_spk('希望你以后能够做的比我还好呦。', prompt_speech_16k, 'my_zero_shot_spk') is True
for i, j in enumerate(cosyvoice.inference_zero_shot('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '', '', zero_shot_spk_id='my_zero_shot_spk', stream=False)):
    torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

cosyvoice.save_spkinfo()

# 指令使用示例
for i, j in enumerate(cosyvoice.inference_instruct2('收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。', '用四川话说这句话', prompt_speech_16k, stream=False)):
    torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
```