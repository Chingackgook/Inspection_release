为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式和参数，并构建相应的 `kwargs` 字典。以下是对每个关键函数的分析和替换方案：

### 1. `download_models`
- **分析**：该函数用于下载模型，参数包括 `source`, `force_redownload`, 和 `custom_path`。
- **参数获取**：
  - `source`: 从 `chat.load()` 中获取，值为 `"huggingface"`。
  - `force_redownload`: 默认值为 `False`。
  - `custom_path`: 该参数在 `load` 方法中未使用，因此可以省略。
- **替换方案**：
  ```python
  exe.run("download_models", source="huggingface", force_redownload=False)
  ```

### 2. `has_loaded`
- **分析**：该函数用于检查模块是否已加载，参数为 `use_decoder`。
- **参数获取**：
  - `use_decoder`: 默认值为 `False`。
- **替换方案**：
  ```python
  exe.run("has_loaded", use_decoder=False)
  ```

### 3. `infer`
- **分析**：该函数用于推理文本，参数包括 `text`, `stream`, `lang`, `skip_refine_text`, `refine_text_only`, `use_decoder`, `do_text_normalization`, `do_homophone_replacement`, `split_text`, `max_split_batch`, `params_refine_text`, 和 `params_infer_code`。
- **参数获取**：
  - `text`: 从 `texts` 列表中获取。
  - `stream`: 默认值为 `False`。
  - `lang`: 未提供，默认为 `None`。
  - `skip_refine_text`: 默认值为 `False`。
  - `refine_text_only`: 设为 `True`。
  - `use_decoder`: 默认值为 `True`。
  - `do_text_normalization`: 默认值为 `True`。
  - `do_homophone_replacement`: 默认值为 `True`。
  - `split_text`: 设为 `False`。
  - `max_split_batch`: 默认值为 `4`。
  - `params_refine_text`: 使用 `ChatTTS.Chat.RefineTextParams(show_tqdm=False)`。
- **替换方案**：
  ```python
  exe.run("infer", text=texts, stream=False, lang=None, skip_refine_text=False, refine_text_only=True, use_decoder=True, do_text_normalization=True, do_homophone_replacement=True, split_text=False, max_split_batch=4, params_refine_text=ChatTTS.Chat.RefineTextParams(show_tqdm=False))
  ```

### 4. `unload`
- **分析**：该函数用于释放已加载的模型和资源，无参数。
- **替换方案**：
  ```python
  exe.run("unload")
  ```

### 5. `sample_random_speaker`
- **分析**：该函数用于随机选择说话者，无参数。
- **替换方案**：
  ```python
  exe.run("sample_random_speaker")
  ```

### 6. `sample_audio_speaker`
- **分析**：该函数用于对输入的语音数据进行编码以获取说话者信息，参数为 `wav`。
- **参数获取**：需要提供一个 `wav` 输入，假设我们有一个 `wav_data` 变量。
- **替换方案**：
  ```python
  exe.run("sample_audio_speaker", wav=wav_data)
  ```

### 7. `interrupt`
- **分析**：该函数用于设置当前上下文为中断状态，无参数。
- **替换方案**：
  ```python
  exe.run("interrupt")
  ```

### 总结
在替换过程中，我们需要确保所有的参数都能正确传递，并且在调用 `exe.run` 时，参数的名称和类型都要与原函数一致。通过这种方式，我们可以将原有的函数调用替换为新的 `exe.run` 调用形式，同时保持功能的完整性。