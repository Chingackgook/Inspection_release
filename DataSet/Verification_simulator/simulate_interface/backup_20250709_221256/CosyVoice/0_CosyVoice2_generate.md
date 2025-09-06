为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，并提供模拟输入的方案，我们可以按照以下步骤进行分析和设计：

### 1. 了解源代码中关键函数的调用

在源代码中，关键函数的调用如下：

- `list_available_spks()`
- `add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)`
- `save_spkinfo()`
- `inference_sft(tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)`
- `inference_zero_shot(tts_text, prompt_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
- `inference_cross_lingual(tts_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
- `inference_instruct(tts_text, spk_id, instruct_text, stream=False, speed=1.0, text_frontend=True)`
- `inference_instruct2(tts_text, instruct_text, prompt_speech_16k, zero_shot_spk_id='', stream=False, speed=1.0, text_frontend=True)`
- `inference_vc(source_speech_16k, prompt_speech_16k, stream=False, speed=1.0)`

### 2. 替换为 `exe.run("function_name", **kwargs)`

我们需要将每个函数的调用替换为 `exe.run("function_name", **kwargs)` 的形式。以下是替换后的示例：

- `cosyvoice.list_available_spks()` 替换为 `exe.run("list_available_spks")`
- `cosyvoice.add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)` 替换为 `exe.run("add_zero_shot_spk", prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, zero_shot_spk_id=zero_shot_spk_id)`
- `cosyvoice.save_spkinfo()` 替换为 `exe.run("save_spkinfo")`
- `cosyvoice.inference_sft(tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)` 替换为 `exe.run("inference_sft", tts_text=tts_text, spk_id=spk_id, stream=stream, speed=speed, text_frontend=text_frontend)`
- `cosyvoice.inference_zero_shot(...)` 替换为 `exe.run("inference_zero_shot", tts_text=tts_text, prompt_text=prompt_text, prompt_speech_16k=prompt_speech_16k, zero_shot_spk_id=zero_shot_spk_id, stream=stream, speed=speed, text_frontend=text_frontend)`
- 其他函数同理。

### 3. 提供模拟输入

对于每个函数，我们需要提供模拟输入。以下是每个函数的参数分析和模拟输入方案：

- **`list_available_spks()`**
  - 无参数，直接调用。

- **`add_zero_shot_spk(prompt_text, prompt_speech_16k, zero_shot_spk_id)`**
  - `prompt_text`: 模拟为 "希望你以后能够做的比我还好呦。"
  - `prompt_speech_16k`: 使用 `load_wav('./asset/zero_shot_prompt.wav', 16000)` 加载的音频数据。
  - `zero_shot_spk_id`: 模拟为 "my_zero_shot_spk"。

- **`save_spkinfo()`**
  - 无参数，直接调用。

- **`inference_sft(tts_text, spk_id, stream=False, speed=1.0, text_frontend=True)`**
  - `tts_text`: 模拟为 "合成的文本示例。"
  - `spk_id`: 模拟为 "my_zero_shot_spk"。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。
  - `text_frontend`: 默认为 `True`。

- **`inference_zero_shot(...)`**
  - `tts_text`: 模拟为 "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
  - `prompt_text`: 模拟为 "希望你以后能够做的比我还好呦。"。
  - `prompt_speech_16k`: 使用 `load_wav('./asset/zero_shot_prompt.wav', 16000)` 加载的音频数据。
  - `zero_shot_spk_id`: 模拟为 "my_zero_shot_spk"。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。
  - `text_frontend`: 默认为 `True`。

- **`inference_cross_lingual(...)`**
  - `tts_text`: 模拟为 "在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。"
  - `prompt_speech_16k`: 使用 `load_wav('./asset/zero_shot_prompt.wav', 16000)` 加载的音频数据。
  - `zero_shot_spk_id`: 默认为空。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。
  - `text_frontend`: 默认为 `True`。

- **`inference_instruct(...)`**
  - `tts_text`: 模拟为 "收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。"
  - `spk_id`: 模拟为 "my_zero_shot_spk"。
  - `instruct_text`: 模拟为 "用四川话说这句话"。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。
  - `text_frontend`: 默认为 `True`。

- **`inference_instruct2(...)`**
  - `tts_text`: 同上。
  - `instruct_text`: 同上。
  - `prompt_speech_16k`: 同上。
  - `zero_shot_spk_id`: 默认为空。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。
  - `text_frontend`: 默认为 `True`。

- **`inference_vc(...)`**
  - `source_speech_16k`: 使用 `load_wav('./asset/source_speech.wav', 16000)` 加载的音频数据。
  - `prompt_speech_16k`: 使用 `load_wav('./asset/prompt_speech.wav', 16000)` 加载的音频数据。
  - `stream`: 默认为 `False`。
  - `speed`: 默认为 `1.0`。

### 总结

通过以上分析，我们可以将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，并为每个函数提供合适的模拟输入。这将有助于在新的执行环境中正确调用这些函数。