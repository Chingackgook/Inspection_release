为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式，并提取出所需的参数。以下是对每个关键函数的分析和替换方案：

### 1. 函数调用分析与替换

#### 1.1 `request_gpt_model_in_new_thread_with_ui_alive`
- **原调用**:
  ```python
  gpt_say = yield from request_gpt_model_in_new_thread_with_ui_alive(i_say, i_say_show_user, llm_kwargs, chatbot, history=[], sys_prompt=system_prompt)
  ```
- **参数**:
  - `inputs`: `i_say`
  - `inputs_show_user`: `i_say_show_user`
  - `llm_kwargs`: `llm_kwargs`
  - `chatbot`: `chatbot`
  - `history`: `[]` (空列表)
  - `sys_prompt`: `system_prompt`
  
- **替换**:
  ```python
  gpt_say = yield from exe.run("request_gpt_model_in_new_thread_with_ui_alive", 
                                inputs=i_say, 
                                inputs_show_user=i_say_show_user, 
                                llm_kwargs=llm_kwargs, 
                                chatbot=chatbot, 
                                history=[], 
                                sys_prompt=system_prompt)
  ```

#### 1.2 `update_ui`
- **原调用**:
  ```python
  yield from update_ui(chatbot=chatbot, history=history)
  ```
- **参数**:
  - `chatbot`: `chatbot`
  - `history`: `history`
  
- **替换**:
  ```python
  yield from exe.run("update_ui", 
                     chatbot=chatbot, 
                     history=history)
  ```

#### 1.3 `write_history_to_file`
- **原调用**:
  ```python
  res = write_history_to_file(history)
  ```
- **参数**:
  - `history`: `history`
  
- **替换**:
  ```python
  res = exe.run("write_history_to_file", 
                history=history)
  ```

#### 1.4 `promote_file_to_downloadzone`
- **原调用**:
  ```python
  promote_file_to_downloadzone(res, chatbot=chatbot)
  ```
- **参数**:
  - `res`: `res`
  - `chatbot`: `chatbot`
  
- **替换**:
  ```python
  exe.run("promote_file_to_downloadzone", 
           res=res, 
           chatbot=chatbot)
  ```

### 2. 模拟输入方案

为了模拟输入，我们需要为每个函数提供合适的参数。以下是参数的模拟方案：

- **`file_manifest`**: 模拟为一个包含文件路径的列表，例如 `["file1.txt", "file2.txt"]`。
- **`project_folder`**: 模拟为一个字符串，表示项目的根目录，例如 `"/path/to/project"`。
- **`llm_kwargs`**: 模拟为一个字典，包含模型参数，例如 `{"model": "gpt-3", "temperature": 0.7}`。
- **`plugin_kwargs`**: 可以为空字典 `{}`，如果没有使用插件。
- **`chatbot`**: 模拟为一个对象，具有 `append` 方法的列表，例如 `[]`。
- **`history`**: 模拟为一个空列表 `[]`，用于存储对话历史。
- **`system_prompt`**: 模拟为一个字符串，例如 `"请分析以下内容"`。

### 3. 总结

通过以上分析，我们可以将源代码中的关键函数调用替换为 `exe.run("function_name", **kwargs)` 的形式，并为每个函数提供合适的模拟输入。这样可以确保代码在新的执行环境中逻辑等价地运行。