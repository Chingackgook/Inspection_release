# Mistral AI 适配器

这是一个用于与 Mistral AI 模型交互的适配器，参照项目的标准适配器设计模式实现。

## 功能特性

- ✅ **文本生成**: 支持多轮对话的文本生成，使用先进的Mistral语言模型
- ✅ **图片处理**: 支持 Pixtral 多模态模型进行图片分析
- ✅ **音频处理**: 提供音频文件信息处理（有限支持）
- ✅ **错误重试**: 内置重试机制，提高稳定性
- ✅ **历史记录**: 支持对话历史管理
- ✅ **Token统计**: 提供详细的token使用量统计

## 支持的模型

### 文本模型
- **mistral-large-latest**: 最新的大型模型，性能强劲
- **mistral-medium-latest**: 中等大小模型，平衡性能与成本
- **mistral-small-latest**: 小型模型，快速响应
- **open-mistral-7b**: 开源7B参数模型
- **open-mixtral-8x7b**: 开源混合专家模型
- **open-mixtral-8x22b**: 更大的混合专家模型

### 多模态模型
- **pixtral-large-latest**: 大型多模态模型，支持图片和文本
- **pixtral-12b-2409**: 12B参数的多模态模型

### 推理模型
- **magistral-small-2506**: 小型推理模型，提供逐步思考能力
- **magistral-medium-2506**: 中等推理模型，更强的分析能力

## 安装依赖

```bash
pip install mistralai pillow
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "mistral_api_key": "your_mistral_api_key_here",
    "mistral_model": "mistral-large-latest",
    "mistral_model_img": "pixtral-large-latest",
    "mistral_model_audio": "mistral-large-latest"
}
```

### 获取 Mistral API 密钥

1. 访问 [Mistral AI La Plateforme](https://console.mistral.ai/)
2. 注册账户并登录
3. 在 API Keys 页面创建新的 API 密钥
4. 将密钥添加到配置文件中

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Mistral_adapter import MistralAdapter

# 初始化适配器
adapter = MistralAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下Mistral AI",
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
    prompt="请详细描述这张图片的内容",
    filepath=["path/to/image.jpg", "path/to/image2.png"],
    temperature=0.3,
    max_tokens=1000
)

print(f"图片分析: {response}")
```

### 音频处理

```python
# 处理音频（当前为文件信息处理）
response = adapter.generate_audio(
    history=history,
    prompt="请分析这些音频文件",
    filepath=["path/to/audio.mp3"],
    temperature=0.3,
    max_tokens=1000
)

print(f"音频处理: {response}")
```

### 在项目中使用

要在项目的 BaseAI 中使用 Mistral 适配器，需要：

1. 在 `config.json` 中设置 `"provider": "Mistral"`
2. 确保已配置正确的 Mistral API 密钥

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载 Mistral 适配器
ai = BaseAI()
response = ai.generate_text("解释一下量子计算的基本原理")
```

## 技术细节

### 温度参数处理
Mistral API 的 temperature 参数范围是 0.0-1.0，适配器会自动处理参数转换：
- 如果传入的 temperature > 1，会自动除以100进行转换
- 例如：传入30会转换为0.3

### 图片格式支持
支持的图片格式：
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

### 对话历史格式
适配器会自动将项目标准的对话历史格式转换为 Mistral API 要求的格式：
- 支持 "user", "assistant", "system" 角色
- 自动处理消息格式转换
