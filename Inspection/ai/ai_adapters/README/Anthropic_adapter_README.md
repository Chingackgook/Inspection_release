# Anthropic Claude 适配器

这是一个用于与 Anthropic Claude AI 模型交互的适配器，参照项目的标准适配器设计模式实现。

## 功能特性

- ✅ **文本生成**: 支持多轮对话的文本生成，使用Claude系列语言模型
- ✅ **图片处理**: 支持 Claude 3+ 的视觉能力进行图片分析
- ✅ **音频处理**: 提供音频文件信息处理（有限支持）
- ✅ **错误重试**: 内置重试机制，提高稳定性
- ✅ **历史记录**: 支持对话历史管理
- ✅ **Token统计**: 提供准确的token使用量统计
- ✅ **安全性**: Claude内置强大的安全防护机制

## 支持的模型

### Claude 4 系列
- **claude-opus-4-20250514**: 最新的旗舰模型，性能最强
- **claude-sonnet-4-20250514**: 平衡性能与速度的最新模型

### Claude 3.7 系列
- **claude-3-7-sonnet-20250219**: 增强版Claude 3.5，更强的推理能力

### Claude 3.5 系列
- **claude-3-5-sonnet-20241022**: 最新的Claude 3.5，高性能模型
- **claude-3-5-sonnet-20240620**: 上一代Claude 3.5版本
- **claude-3-5-haiku-20241022**: 轻量级高速模型

### Claude 3 系列
- **claude-3-opus-20240229**: 原始旗舰模型，最强性能
- **claude-3-sonnet-20240229**: 平衡版本
- **claude-3-haiku-20240307**: 最快的轻量级模型

## 模型选择建议

| 使用场景 | 推荐模型 | 特点 |
|----------|----------|------|
| 复杂推理、创意写作 | claude-opus-4-20250514 | 最强智能，最高质量 |
| 日常对话、代码编写 | claude-sonnet-4-20250514 | 性能与速度平衡 |
| 快速响应、简单任务 | claude-3-5-haiku-20241022 | 速度最快，成本最低 |
| 图片分析 | claude-3-5-sonnet-20241022+ | 优秀的视觉能力 |

## 安装依赖

```bash
pip install anthropic pillow
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "anthropic_api_key": "your_anthropic_api_key_here",
    "anthropic_model": "claude-3-5-sonnet-20241022",
    "anthropic_model_img": "claude-3-5-sonnet-20241022",
    "anthropic_model_audio": "claude-3-5-sonnet-20241022"
}
```

### 获取 Anthropic API 密钥

1. 访问 [Anthropic Console](https://console.anthropic.com/)
2. 注册账户并登录
3. 在 API Keys 页面创建新的 API 密钥
4. 将密钥添加到配置文件中

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Anthropic_adapter import AnthropicAdapter

# 初始化适配器
adapter = AnthropicAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下Claude AI",
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
    prompt="请详细描述这张图片的内容，包括颜色、对象和场景",
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
    prompt="请分析这些音频文件的基本信息",
    filepath=["path/to/audio.mp3"],
    temperature=0.3,
    max_tokens=1000
)

print(f"音频处理: {response}")
```

### 在项目中使用

要在项目的 BaseAI 中使用 Anthropic 适配器，需要：

1. 在 `config.json` 中设置 `"provider": "Anthropic"`
2. 确保已配置正确的 Anthropic API 密钥

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载 Anthropic 适配器
ai = BaseAI()
response = ai.generate_text("解释一下人工智能的伦理问题")
```

## 技术细节

### 温度参数处理
Anthropic API 的 temperature 参数范围是 0.0-1.0，适配器会自动处理参数转换：
- 如果传入的 temperature > 1，会自动除以100进行转换
- 例如：传入30会转换为0.3

### 图片格式支持
支持的图片格式：
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

图片会被自动转换为base64格式并按照Anthropic的格式要求发送。

### 对话历史格式
适配器会自动将项目标准的对话历史格式转换为 Anthropic API 要求的格式：
- 支持 "user", "assistant" 角色
- 自动过滤 "system" 角色（Anthropic使用单独的system参数）
- 保持对话的连贯性和上下文

### 响应处理
适配器会自动处理Claude的响应格式：
- 提取文本内容从response.content
- 处理多种内容类型
- 提供准确的token统计信息

## 与其他适配器的对比

| 功能 | OpenAI | Google | Mistral | Anthropic |
|------|--------|--------|---------|-----------|
| 文本生成 | ✅ | ✅ | ✅ | ✅ |
| 图片处理 | ✅ | ✅ | ✅ | ✅ |
| 音频处理 | 部分支持 | ✅ | 有限支持 | 有限支持 |
| Token 统计 | 精确 | 估算 | 精确 | 精确 |
| 安全性 | 中等 | 中等 | 中等 | 最高 |
| 推理能力 | 强 | 强 | 强 | 最强 |
| 创意写作 | 好 | 好 | 好 | 优秀 |
| 代码生成 | 优秀 | 好 | 好 | 优秀 |

## 错误处理

适配器内置了完善的错误处理机制：

- **自动重试**: 最多重试5次，提高稳定性
- **详细日志**: 提供详细的错误信息和调试信息
- **优雅降级**: 在某些功能不可用时提供替代方案
- **API错误**: 自动处理常见的API错误和限制

## 注意事项

1. **API 配额**: Anthropic 有使用配额限制，请注意监控使用量
2. **网络连接**: 需要稳定的网络连接访问 Anthropic 服务
3. **文件大小**: 图片文件有大小限制，建议压缩大文件
4. **音频支持**: 当前版本不支持直接音频处理
5. **系统消息**: Anthropic的系统消息处理方式与其他API不同
6. **响应格式**: Claude的响应格式经过特别优化，通常更加结构化

## 最佳实践

### 1. 模型选择策略
```python
# 根据任务复杂度选择模型
simple_tasks = "claude-3-5-haiku-20241022"     # 快速任务
balanced_tasks = "claude-3-5-sonnet-20241022"  # 平衡任务  
complex_tasks = "claude-opus-4-20250514"       # 复杂任务
```

### 2. 温度设置建议
```python
creative_temp = 0.7    # 创意写作
analytical_temp = 0.2  # 分析任务
factual_temp = 0.1     # 事实问答
```

### 3. 上下文管理
- 合理控制对话历史长度
- 重要信息可以重复强调
- 使用清晰的指令和结构

### 4. 安全使用
- 避免尝试绕过安全机制
- 遵循Anthropic的使用政策
- 注意输出内容的审查
