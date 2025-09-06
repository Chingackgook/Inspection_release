# 接口文档

## 2. 函数说明

### 2.1. `request_gpt_model_in_new_thread_with_ui_alive`

#### 函数名
`request_gpt_model_in_new_thread_with_ui_alive`

#### 参数说明
- `inputs` (string): 输入的文本内容。
- `inputs_show_user` (string): 展示给用户的输入文本。
- `llm_kwargs` (dict): GPT模型的参数设置。
- `chatbot` (object): 用户界面的对话窗口句柄，用于数据流可视化。
- `history` (list): 对话历史记录列表。
- `sys_prompt` (string): 系统提示信息。
- `refresh_interval` (float, optional): UI 刷新时间间隔，默认为 0.2 秒。
- `handle_token_exceed` (bool, optional): 是否自动处理 token 溢出，默认为 True。
- `retry_times_at_unknown_error` (int, optional): 遇到未知错误时的重试次数，默认为 2。

#### 返回值说明
- `future`: GPT模型返回的结果，类型为 `Future` 对象。

#### 范围说明
- `inputs` 和 `inputs_show_user` 应为字符串类型，且长度应适合模型处理。
- `llm_kwargs` 应包含有效的模型参数。
- `chatbot` 应为有效的用户界面对象。
- `history` 应为有效的对话历史列表。
- `sys_prompt` 应为有效的系统提示字符串。
- `refresh_interval` 应在 0.1 到 3 之间。
- `handle_token_exceed` 和 `retry_times_at_unknown_error` 应为布尔值和整数。

---

## 3. 其他相关函数说明

### 3.1. `predict_no_ui_long_connection`

#### 函数名
`predict_no_ui_long_connection`

#### 参数说明
- `inputs` (string): 输入的文本内容。
- `llm_kwargs` (dict): GPT模型的参数设置。
- `history` (list): 对话历史记录列表。
- `sys_prompt` (string): 系统提示信息。
- `observe_window` (list): 用于监控状态的可变列表。

#### 返回值说明
- `result`: GPT模型的输出结果，类型为字符串。

#### 范围说明
- `inputs` 应为字符串类型，且长度应适合模型处理。
- `llm_kwargs` 应包含有效的模型参数。
- `history` 应为有效的对话历史列表。
- `sys_prompt` 应为有效的系统提示字符串。
- `observe_window` 应为可变列表，包含监控状态的信息。

---

### 3.2. `update_ui`

#### 函数名
`update_ui`

#### 参数说明
- `chatbot` (object): 用户界面的对话窗口句柄。
- `history` (list): 对话历史记录列表。

#### 返回值说明
- `None`: 此函数不返回任何值。

#### 范围说明
- `chatbot` 应为有效的用户界面对象。
- `history` 应为有效的对话历史列表。

---

### 3.3. `get_conf`

#### 函数名
`get_conf`

#### 参数说明
- `key` (string): 配置项的键。

#### 返回值说明
- `value`: 配置项的值，类型取决于配置项。

#### 范围说明
- `key` 应为有效的字符串，表示配置项的名称。

---

### 3.4. `trimmed_format_exc`

#### 函数名
`trimmed_format_exc`

#### 返回值说明
- `str`: 格式化的异常信息字符串。

#### 范围说明
- 此函数不接受参数，返回当前异常的格式化字符串。

---

### 3.5. `get_max_token`

#### 函数名
`get_max_token`

#### 参数说明
- `llm_kwargs` (dict): GPT模型的参数设置。

#### 返回值说明
- `int`: 模型允许的最大 token 数量。

#### 范围说明
- `llm_kwargs` 应包含有效的模型参数。