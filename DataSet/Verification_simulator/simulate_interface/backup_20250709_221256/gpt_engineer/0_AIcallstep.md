为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐步分析每个函数的调用方式，并确定如何构造 `kwargs` 参数。以下是对每个关键函数的分析和替换方案：

### 1. AI 类的关键函数

#### AI_start
- **调用示例**: `messages = ai.start(system, user, step_name)`
- **替换方案**: 
  - `exe.run("AI_start", system=system, user=user, step_name=step_name)`
- **参数获取**:
  - `system`: 从 `setup_sys_prompt(preprompts)` 获取。
  - `user`: 通过 `prompt.to_langchain_content()` 获取。
  - `step_name`: 使用 `curr_fn()` 获取。

#### AI_next
- **调用示例**: `updated_messages = ai.next(messages, prompt, step_name)`
- **替换方案**: 
  - `exe.run("AI_next", messages=messages, prompt=prompt, step_name=step_name)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。
  - `prompt`: 直接传入。
  - `step_name`: 使用 `curr_fn()` 获取。

#### AI_backoff_inference
- **调用示例**: `response = ai.backoff_inference(messages)`
- **替换方案**: 
  - `exe.run("AI_backoff_inference", messages=messages)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。

#### AI_serialize_messages
- **调用示例**: `json_string = ai.serialize_messages(messages)`
- **替换方案**: 
  - `exe.run("AI_serialize_messages", messages=messages)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。

#### AI_deserialize_messages
- **调用示例**: `messages = ai.deserialize_messages(json_string)`
- **替换方案**: 
  - `exe.run("AI_deserialize_messages", jsondictstr=json_string)`
- **参数获取**:
  - `json_string`: 直接使用之前序列化的 JSON 字符串。

#### AI__create_chat_model
- **调用示例**: `chat_model = ai._create_chat_model()`
- **替换方案**: 
  - `exe.run("AI__create_chat_model")`
- **参数获取**: 无参数。

### 2. ClipboardAI 类的关键函数

#### ClipboardAI_serialize_messages
- **调用示例**: `serialized = clipboard_ai_instance.serialize_messages(messages)`
- **替换方案**: 
  - `exe.run("ClipboardAI_serialize_messages", messages=messages)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。

#### ClipboardAI_multiline_input
- **调用示例**: `user_input = clipboard_ai_instance.multiline_input()`
- **替换方案**: 
  - `exe.run("ClipboardAI_multiline_input")`
- **参数获取**: 无参数。

#### ClipboardAI_next
- **调用示例**: `updated_messages = clipboard_ai_instance.next(messages, prompt, step_name)`
- **替换方案**: 
  - `exe.run("ClipboardAI_next", messages=messages, prompt=prompt, step_name=step_name)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。
  - `prompt`: 直接传入。
  - `step_name`: 使用 `curr_fn()` 获取。

### 3. serialize_messages 函数
- **调用示例**: `json_string = serialize_messages(messages)`
- **替换方案**: 
  - `exe.run("serialize_messages", messages=messages)`
- **参数获取**:
  - `messages`: 直接使用之前的消息列表。

### 总结
在替换过程中，我们需要确保每个函数的参数都能正确传递。对于需要外部输入的参数，我们可以通过模拟输入或从上下文中提取值来获取。最终的替换方案将使得所有关键函数调用都通过 `exe.run` 进行，从而实现逻辑等价执行。