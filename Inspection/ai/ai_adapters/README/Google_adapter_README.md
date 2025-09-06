# Google API 适配器

这是一个用于与 Google Gemini AI 模型交互的适配器，参照 OpenAI_adapter.py 的设计模式实现。

## 功能特性

- ✅ **文本生成**: 支持多轮对话的文本生成
- ✅ **图片处理**: 支持上传图片并进行分析
- ✅ **音频处理**: 支持音频文件的处理和分析
- ✅ **错误重试**: 内置重试机制，提高稳定性
- ✅ **历史记录**: 支持对话历史管理

## 安装依赖

```bash
pip install google-generativeai pillow
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "google_api_key": "your_google_api_key_here",
    "google_model": "gemini-1.5-flash",
    "google_model_img": "gemini-1.5-flash", 
    "google_model_audio": "gemini-1.5-flash"
}
```

### 获取 Google API 密钥

1. 访问 [Google AI Studio](https://aistudio.google.com/app/apikey)
2. 创建新的 API 密钥
3. 将密钥添加到配置文件中

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Google_adapter import GoogleAdapter

# 初始化适配器
adapter = GoogleAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下你自己",
    temperature=0.3,
    max_tokens=1000
)

print(f"回复: {response}")
print(f"Token使用量: {token_usage}")
```

### 图片分析

```python
# 分析图片
response = adapter.generate_image(
    history=history,
    prompt="请描述这张图片的内容",
    filepath=["path/to/image.jpg"],
    temperature=0.3,
    max_tokens=1000
)

print(f"图片分析: {response}")
```

### 音频处理

```python
# 处理音频
response = adapter.generate_audio(
    history=history,
    prompt="请分析这个音频的内容",
    filepath=["path/to/audio.mp3"],
    temperature=0.3,
    max_tokens=1000
)

print(f"音频分析: {response}")
```

### 在项目中使用

要在项目的 BaseAI 中使用 Google 适配器，需要：

1. 在 `config.json` 中设置 `"provider": "Google"`
2. 确保已配置正确的 Google API 密钥

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载 Google 适配器
ai = BaseAI()
response = ai.generate_text("你好，世界！")
```

## 支持的模型

- **gemini-1.5-flash**: 快速响应的多模态模型
- **gemini-1.5-pro**: 高质量的多模态模型
- **gemini-1.0-pro**: 文本专用模型

## 注意事项

1. **API 配额**: Google AI 有使用配额限制，请注意监控使用量
2. **文件大小**: 图片和音频文件有大小限制
3. **支持格式**: 
   - 图片: JPEG, PNG, GIF, WebP
   - 音频: MP3, WAV, FLAC, AAC, OGG
4. **网络连接**: 需要稳定的网络连接访问 Google AI 服务

## 错误处理

适配器内置了重试机制和错误处理：

- 自动重试最多 5 次
- 详细的错误日志输出
- 友好的错误提示信息
