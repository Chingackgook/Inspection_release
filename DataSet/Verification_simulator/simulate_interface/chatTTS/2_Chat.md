为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式和参数，并构建相应的 `kwargs` 字典。以下是对每个关键函数的分析和替换方案：

### 1. `download_models`
- **调用分析**：该函数用于下载模型，通常需要指定下载源、是否强制重新下载等参数。
- **参数**：
  - `source`: 可能的值为 "huggingface", "local", "custom"。
  - `force_redownload`: 布尔值，指示是否强制重新下载。
  - `custom_path`: 自定义路径（可选）。
- **替换方案**：
  ```python
  exe.run("download_models", source="huggingface", force_redownload=False)
  ```

### 2. `has_loaded`
- **调用分析**：该函数用于检查模块是否已加载，通常不需要额外参数。
- **参数**：
  - `use_decoder`: 布尔值，指示是否检查解码器模块。
- **替换方案**：
  ```python
  exe.run("has_loaded", use_decoder=False)
  ```

### 3. `infer`
- **调用分析**：该函数用于推理文本，返回推理结果。需要传入文本和其他推理参数。
- **参数**：
  - `text`: 需要推理的文本（字符串或字符串列表）。
  - `refine_text_only`: 布尔值，指示是否只返回精炼后的文本。
  - `params_refine_text`: 需要传入的参数对象。
  - 其他推理参数（如 `stream`, `lang`, `skip_refine_text` 等）可以根据需要添加。
- **替换方案**：
  ```python
  refined_text = exe.run("infer", text=text, refine_text_only=True, params_refine_text=params_refine_text)
  ```

### 4. `unload`
- **调用分析**：该函数用于释放已加载的模型和资源，通常不需要参数。
- **替换方案**：
  ```python
  exe.run("unload")
  ```

### 5. `sample_random_speaker`
- **调用分析**：该函数用于随机选择说话者，通常不需要参数。
- **替换方案**：
  ```python
  rand_spk = exe.run("sample_random_speaker")
  ```

### 6. `sample_audio_speaker`
- **调用分析**：该函数用于对输入的语音数据进行编码以获取说话者信息，通常需要传入音频数据。
- **参数**：
  - `wav`: 输入的音频数据（numpy.ndarray 或 torch.Tensor）。
- **替换方案**：
  ```python
  encoded_speaker = exe.run("sample_audio_speaker", wav=audio_data)
  ```

### 7. `interrupt`
- **调用分析**：该函数用于设置当前上下文为中断状态，通常不需要参数。
- **替换方案**：
  ```python
  exe.run("interrupt")
  ```

### 模拟输入方案
在替换函数调用时，我们需要确保所有参数都能正确传递。以下是对每个函数的模拟输入方案：

- **`download_models`**: 使用默认参数或根据需要指定 `source` 和 `force_redownload`。
- **`has_loaded`**: 使用默认参数。
- **`infer`**: 
  - `text`: 使用源代码中的 `text` 变量。
  - `params_refine_text`: 创建一个 `RefineTextParams` 对象，使用源代码中的参数。
- **`unload`**: 无需参数。
- **`sample_random_speaker`**: 无需参数。
- **`sample_audio_speaker`**: 需要提供音频数据，可能需要在代码中定义或模拟。
- **`interrupt`**: 无需参数。

### 总结
通过以上分析，我们可以将源代码中的关键函数调用替换为 `exe.run("function_name", **kwargs)` 的形式，并确保所有参数都能正确传递。这样可以使代码更加模块化和可维护。