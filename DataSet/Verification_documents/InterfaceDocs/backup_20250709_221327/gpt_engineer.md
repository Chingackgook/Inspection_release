# 接口文档

## 类：AI

### 初始化方法：`__init__`
- **函数名**: `__init__`
- **参数说明**:
  - `model_name` (str, optional): 使用的模型名称，默认为 "gpt-4-turbo"。
  - `temperature` (float, optional): 模型的温度设置，默认为 0.1。
  - `azure_endpoint` (str, optional): Azure 托管语言模型的端点 URL，默认为 None。
  - `streaming` (bool, optional): 是否使用流式处理，默认为 True。
  - `vision` (bool, optional): 是否启用视觉功能，默认为 False。
- **返回值说明**: 无返回值。
- **范围说明**: 初始化 AI 类的实例。

### 属性
- `temperature` (float): 语言模型的温度设置。
- `azure_endpoint` (str): Azure 托管语言模型的端点 URL。
- `model_name` (str): 使用的语言模型名称。
- `streaming` (bool): 是否使用流式处理。
- `llm` (BaseChatModel): 语言模型实例，用于会话管理。
- `token_usage_log` (TokenUsageLog): 用于跟踪会话中的令牌使用情况的日志。

### 方法：`start`
- **函数名**: `start`
- **参数说明**:
  - `system` (str): 系统消息的内容。
  - `user` (Any): 用户消息的内容。
  - `step_name` (str): 步骤名称。
- **返回值说明**: `List[Message]` - 会话中的消息列表。
- **范围说明**: 启动会话，返回初始消息列表。

#### 调用示例
```python
ai_instance = AI()
messages = ai_instance.start("系统消息", "用户消息", step_name="步骤1")
```

### 方法：`next`
- **函数名**: `next`
- **参数说明**:
  - `messages` (List[Message]): 会话中的消息列表。
  - `prompt` (Optional[str], optional): 使用的提示，默认为 None。
  - `step_name` (str): 步骤名称。
- **返回值说明**: `List[Message]` - 更新后的会话消息列表。
- **范围说明**: 通过发送消息历史到 LLM 来推进会话，并更新响应。

#### 调用示例
```python
updated_messages = ai_instance.next(messages, prompt="下一步是什么？", step_name="步骤2")
```

### 方法：`backoff_inference`
- **函数名**: `backoff_inference`
- **参数说明**:
  - `messages` (List[Message]): 将传递给语言模型进行处理的消息列表。
- **返回值说明**: `Any` - 语言模型处理后返回的输出。
- **范围说明**: 使用指数退避策略执行推理。

#### 调用示例
```python
response = ai_instance.backoff_inference(messages)
```

### 方法：`serialize_messages`
- **函数名**: `serialize_messages`
- **参数说明**:
  - `messages` (List[Message]): 要序列化的消息列表。
- **返回值说明**: `str` - 序列化后的消息 JSON 字符串。
- **范围说明**: 将消息列表序列化为 JSON 字符串。

#### 调用示例
```python
json_string = ai_instance.serialize_messages(messages)
```

### 方法：`deserialize_messages`
- **函数名**: `deserialize_messages`
- **参数说明**:
  - `jsondictstr` (str): 要反序列化的 JSON 字符串。
- **返回值说明**: `List[Message]` - 反序列化后的消息列表。
- **范围说明**: 将 JSON 字符串反序列化为消息列表。

#### 调用示例
```python
messages = ai_instance.deserialize_messages(json_string)
```

### 方法：`_create_chat_model`
- **函数名**: `_create_chat_model`
- **参数说明**: 无参数。
- **返回值说明**: `BaseChatModel` - 创建的聊天模型。
- **范围说明**: 创建具有指定模型名称和温度的聊天模型。

#### 调用示例
```python
chat_model = ai_instance._create_chat_model()
```

---

## 类：ClipboardAI

### 初始化方法：`__init__`
- **函数名**: `__init__`
- **参数说明**: 无参数。
- **返回值说明**: 无返回值。
- **范围说明**: 初始化 ClipboardAI 类的实例。

### 方法：`serialize_messages`
- **函数名**: `serialize_messages`
- **参数说明**:
  - `messages` (List[Message]): 要序列化的消息列表。
- **返回值说明**: `str` - 序列化后的消息字符串。
- **范围说明**: 将消息列表序列化为字符串格式。

#### 调用示例
```python
clipboard_ai_instance = ClipboardAI()
serialized = clipboard_ai_instance.serialize_messages(messages)
```

### 方法：`multiline_input`
- **函数名**: `multiline_input`
- **参数说明**: 无参数。
- **返回值说明**: `str` - 用户输入的多行字符串。
- **范围说明**: 处理用户的多行输入。

#### 调用示例
```python
user_input = clipboard_ai_instance.multiline_input()
```

### 方法：`next`
- **函数名**: `next`
- **参数说明**:
  - `messages` (List[Message]): 会话中的消息列表。
  - `prompt` (Optional[str], optional): 使用的提示，默认为 None。
  - `step_name` (str): 步骤名称。
- **返回值说明**: `List[Message]` - 更新后的会话消息列表。
- **范围说明**: 通过将消息复制到剪贴板并处理用户输入来推进会话。

#### 调用示例
```python
updated_messages = clipboard_ai_instance.next(messages, prompt="请继续", step_name="步骤3")
```

---

## 函数：`serialize_messages`
- **函数名**: `serialize_messages`
- **参数说明**:
  - `messages` (List[Message]): 要序列化的消息列表。
- **返回值说明**: `str` - 序列化后的消息 JSON 字符串。
- **范围说明**: 将消息列表序列化为 JSON 字符串。