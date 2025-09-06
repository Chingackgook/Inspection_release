# 接口文档

## 类: OpenAIVisionClient
`OpenAIVisionClient` 类用于与 OpenAI 视觉模型进行交互，继承自 `BaseLLMModel`。该类提供生成回答的核心功能，支持一次性获取完整回答和流式迭代获取部分回答。

### 初始化方法: `__init__(model_name, api_key, user_name="")`
- **参数说明**:
  - `model_name` (str): 模型名称。
  - `api_key` (str): OpenAI API 密钥。
  - `user_name` (str, 可选): 用户名称，默认为空字符串。
  
- **返回值**: 无
- **范围说明**: 初始化 `OpenAIVisionClient` 实例，设置 API 相关的 URL 和请求头。

### 方法: `get_answer_stream_iter()`
- **参数说明**: 无
- **返回值**: 生成器，逐步返回模型的回答内容。
- **范围说明**: 通过流式方式获取模型的回答，支持部分内容的实时返回。

### 方法: `get_answer_at_once()`
- **参数说明**: 无
- **返回值**: tuple: 
  - `content` (str): 模型生成的完整回答内容。
  - `total_token_count` (int): 使用的总 token 数量。
- **范围说明**: 一次性获取模型的完整回答。

### 方法: `count_token(user_input)`
- **参数说明**:
  - `user_input` (str): 用户输入的内容。
  
- **返回值**: int: 计算的 token 数量。
- **范围说明**: 计算用户输入及系统提示的 token 数量。

### 方法: `count_image_tokens(width: int, height: int)`
- **参数说明**:
  - `width` (int): 图像的宽度。
  - `height` (int): 图像的高度。
  
- **返回值**: int: 计算的图像 token 数量。
- **范围说明**: 根据图像的宽度和高度计算所需的 token 数量。

### 方法: `billing_info()`
- **参数说明**: 无
- **返回值**: str: 本月使用金额的 HTML 格式信息。
- **范围说明**: 获取当前用户的 API 使用情况和账单信息。

### 方法: `_get_gpt4v_style_history()`
- **参数说明**: 无
- **返回值**: list: 格式化后的历史对话记录。
- **范围说明**: 获取符合 GPT-4V 风格的历史对话记录。

### 方法: `_get_response(stream=False)`
- **参数说明**:
  - `stream` (bool, 可选): 是否以流式方式获取响应，默认为 False。
  
- **返回值**: Response: HTTP 响应对象。
- **范围说明**: 向 OpenAI API 发送请求并获取响应。

### 方法: `_refresh_header()`
- **参数说明**: 无
- **返回值**: 无
- **范围说明**: 刷新请求头信息。

### 方法: `_get_billing_data(billing_url)`
- **参数说明**:
  - `billing_url` (str): 账单数据的 URL。
  
- **返回值**: dict: 账单数据的 JSON 格式内容。
- **范围说明**: 获取指定 URL 的账单数据。

### 方法: `_decode_chat_response(response)`
- **参数说明**:
  - `response` (Response): HTTP 响应对象。
  
- **返回值**: 生成器，逐步返回解析后的聊天响应内容。
- **范围说明**: 解码聊天响应，处理流式数据。

### 方法: `set_key(new_access_key)`
- **参数说明**:
  - `new_access_key` (str): 新的 API 密钥。
  
- **返回值**: bool: 设置密钥是否成功。
- **范围说明**: 更新 API 密钥并刷新请求头。

### 方法: `_single_query_at_once(history, temperature=1.0)`
- **参数说明**:
  - `history` (list): 历史对话记录。
  - `temperature` (float, 可选): 生成回答的温度，默认为 1.0。
  
- **返回值**: Response: HTTP 响应对象。
- **范围说明**: 发送单次查询请求并获取响应。

### 方法: `auto_name_chat_history(name_chat_method, user_question, single_turn_checkbox)`
- **参数说明**:
  - `name_chat_method` (str): 命名聊天历史的方法。
  - `user_question` (str): 用户提问内容。
  - `single_turn_checkbox` (bool): 单轮对话的复选框状态。
  
- **返回值**: 更新后的状态。
- **范围说明**: 自动命名聊天历史记录，支持多种命名方式。