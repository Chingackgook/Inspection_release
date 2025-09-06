# 接口文档

## 类: `Bridge`

### 初始化方法: `__init__()`
- **参数**: 无
- **返回值**: 无
- **范围**: 
  - 初始化桥接类实例，配置模块类型。
  - 创建 `self.btype` 字典，存储不同功能模块的类型。
  - 从配置文件中获取模型类型以初始化聊天机器人类型。

### 方法: `get_bot(typename)`
- **参数**: 
  - `typename` (str): 需要获取的机器人类型，可能的值包括 "chat", "voice_to_text", "text_to_voice", "translate"。
- **返回值**: `Bot` 实例 (或其子类)
- **范围**: 
  - 根据 `typename` 创建并返回对应的机器人实例。
  - 如果实例已经存在，则直接返回已创建的实例。

### 方法: `get_bot_type(typename)`
- **参数**: 
  - `typename` (str): 需要获取类型的机器人名称。
- **返回值**: (str): 机器人类型
- **范围**: 
  - 获取指定机器人的类型信息。

### 方法: `fetch_reply_content(query, context)`
- **参数**: 
  - `query` (str): 用户查询内容。
  - `context` (Context): 上下文信息，提供给聊天机器人使用。
- **返回值**: `Reply` 实例
- **范围**: 
  - 调用聊天机器人接口获取回复内容。

### 方法: `fetch_voice_to_text(voiceFile)`
- **参数**: 
  - `voiceFile` (str): 音频文件路径，用于语音转文本。
- **返回值**: `Reply` 实例
- **范围**: 
  - 调用语音转文本机器人将语音文件转换为文本。

### 方法: `fetch_text_to_voice(text)`
- **参数**: 
  - `text` (str): 需要转换为语音的文本。
- **返回值**: `Reply` 实例
- **范围**: 
  - 调用文本转语音机器人将文本转换为音频。

### 方法: `fetch_translate(text, from_lang="", to_lang="en")`
- **参数**: 
  - `text` (str): 需要翻译的文本。
  - `from_lang` (str): 源语言，默认值为 "" (自动检测)。
  - `to_lang` (str): 目标语言，默认值为 "en"。
- **返回值**: `Reply` 实例
- **范围**: 
  - 调用翻译机器人进行文本翻译。

### 方法: `find_chat_bot(bot_type)`
- **参数**:
  - `bot_type` (str): 需要查找的聊天机器人类型。
- **返回值**: `Bot` 实例 (或其子类)
- **范围**: 
  - 查找并返回指定类型的聊天机器人实例。

### 方法: `reset_bot()`
- **参数**: 无
- **返回值**: 无
- **范围**: 
  - 重置机器人路由，重新初始化 `Bridge` 类的实例。

## 示例调用
```python
# 创建桥接实例
bridge = Bridge()

# 获取聊天机器人
chat_bot = bridge.get_bot("chat")
reply = bridge.fetch_reply_content("Hello, how are you?", context)

# 语音转文本
voice_text = bridge.fetch_voice_to_text("path/to/audio/file.wav")

# 文本转语音
audio = bridge.fetch_text_to_voice("Hello, world!")

# 翻译文本
translated_text = bridge.fetch_translate("Bonjour", from_lang="fr", to_lang="en")

# 查找特定聊天机器人
specific_bot = bridge.find_chat_bot("chatbot_type")

# 重置机器人
bridge.reset_bot()
``` 

上述接口文档详细描述了 `Bridge` 类中的所有方法及其功能，为开发者提供了清晰的接口说明和示例调用。