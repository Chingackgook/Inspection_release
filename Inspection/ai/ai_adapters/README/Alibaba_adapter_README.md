# 阿里巴巴通义千问适配器

这是一个用于与阿里巴巴通义千问（Qwen）系列模型交互的适配器，基于DashScope API和OpenAI兼容接口实现。

## 功能特性

- ✅ **文本生成**: 支持通义千问全系列文本模型
- ✅ **图片处理**: 支持通义千问-VL多模态模型进行图片分析
- ✅ **音频处理**: 提供音频文件信息处理（有限支持）
- ✅ **错误重试**: 内置重试机制，提高稳定性
- ✅ **历史记录**: 支持对话历史管理
- ✅ **Token统计**: 精确的token使用量统计
- ✅ **OpenAI兼容**: 使用OpenAI SDK调用，迁移成本低
- ✅ **中文优化**: 针对中文语境深度优化

## 支持的模型

### 旗舰商用模型

#### 文本模型
- **qwen-max**: 超大规模语言模型，具备更强的通用能力
- **qwen-plus**: 增强版大语言模型，性能和效果全面升级  
- **qwen-turbo**: 超大规模语言模型，平衡了性能与效率
- **qwen-long**: 长文本模型，支持长达1000万tokens的上下文

#### 多模态模型
- **qwen-vl-max**: 视觉理解能力最强的多模态大模型
- **qwen-vl-plus**: 通用场景效果最佳的多模态大模型

#### 专业模型
- **qwen-coder-plus**: 代码生成和理解能力强化版本
- **qwen-math-plus**: 数学推理专用模型
- **qwen2.5-coder-32b-instruct**: 32B代码专用模型

### 开源模型系列

#### Qwen3 系列（最新）
- **qwen3-32b**: 32B参数开源模型
- **qwen3-14b**: 14B参数平衡版本
- **qwen3-7b**: 7B参数轻量版本
- **qwen3-1.8b**: 1.8B参数超轻量版本
- **qwq-32b**: 专门针对推理优化的32B模型

#### Qwen2.5 系列
- **qwen2.5-72b-instruct**: 72B参数指令微调版本
- **qwen2.5-32b-instruct**: 32B参数指令微调版本
- **qwen2.5-14b-instruct**: 14B参数指令微调版本
- **qwen2.5-7b-instruct**: 7B参数指令微调版本
- **qwen2.5-3b-instruct**: 3B参数轻量版本
- **qwen2.5-1.5b-instruct**: 1.5B参数超轻量版本
- **qwen2.5-0.5b-instruct**: 0.5B参数极轻量版本

#### 专业开源模型
- **qwen2.5-coder-32b-instruct**: 代码生成专用32B模型
- **qwen2.5-coder-14b-instruct**: 代码生成专用14B模型
- **qwen2.5-coder-7b-instruct**: 代码生成专用7B模型
- **qwen2.5-math-72b-instruct**: 数学推理专用72B模型
- **qwen2.5-math-7b-instruct**: 数学推理专用7B模型

## 模型选择建议

| 使用场景 | 推荐模型 | 特点 |
|----------|----------|------|
| 复杂推理、创意写作 | qwen-max | 最强能力，复杂任务首选 |
| 日常对话、通用任务 | qwen-plus | 性能与成本平衡 |
| 快速响应、简单任务 | qwen-turbo | 速度快，成本低 |
| 长文档处理 | qwen-long | 支持1000万tokens上下文 |
| 图片分析 | qwen-vl-max | 最强视觉理解能力 |
| 代码编程 | qwen-coder-plus | 代码生成专家 |
| 数学计算 | qwen-math-plus | 数学推理专用 |
| 开源部署 | qwen3-32b | 最新开源旗舰 |

## 安装依赖

```bash
pip install openai pillow
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "alibaba_api_key": "your_alibaba_api_key_here",
    "alibaba_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "alibaba_model": "qwen-plus",
    "alibaba_model_img": "qwen-vl-plus",
    "alibaba_model_audio": "qwen-plus"
}
```

### 获取阿里云API密钥

1. 访问 [阿里云百炼控制台](https://bailian.console.aliyun.com/)
2. 登录阿里云账号
3. 在API-Key管理页面创建新的API密钥
4. 将密钥添加到配置文件中

### 地域选择

根据你的使用地区选择合适的base_url：

- **中国大陆**: `https://dashscope.aliyuncs.com/compatible-mode/v1`
- **新加坡**: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **金融云**: `https://dashscope-finance.aliyuncs.com/compatible-mode/v1`

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Alibaba_adapter import AlibabaAdapter

# 初始化适配器
adapter = AlibabaAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下通义千问",
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
    prompt="请详细描述这张图片的内容，包括人物、物体、场景等",
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

要在项目的 BaseAI 中使用 Alibaba 适配器，需要：

1. 在 `config.json` 中设置 `"provider": "Alibaba"`
2. 确保已配置正确的阿里云API密钥

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载 Alibaba 适配器
ai = BaseAI()
response = ai.generate_text("用中文解释一下人工智能的发展历程")
```

## 技术细节

### OpenAI兼容性
本适配器使用OpenAI SDK通过DashScope的OpenAI兼容接口调用通义千问模型：
- 完全兼容OpenAI的chat completions接口
- 支持流式和非流式响应
- 支持多模态输入（文本+图片）

### 温度参数处理
温度参数会自动归一化到0.0-1.0范围：
- 如果传入的 temperature > 1，会自动除以100进行转换
- 例如：传入30会转换为0.3

### 图片格式支持
支持的图片格式：
- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

图片会被自动转换为base64格式并按照OpenAI兼容格式发送。

### 对话历史格式
适配器自动处理对话历史格式转换：
- 支持 "user", "assistant", "system" 角色
- 自动转换为OpenAI兼容的消息格式
- 保持对话的连贯性和上下文

## 错误处理

适配器内置了完善的错误处理机制：

- **自动重试**: 最多重试5次，提高稳定性
- **详细日志**: 提供详细的错误信息和调试信息
- **优雅降级**: 在某些功能不可用时提供替代方案
- **网络错误**: 自动处理网络超时和连接错误

## 注意事项

1. **API配额**: 阿里云有使用配额限制，请注意监控使用量
2. **网络连接**: 需要稳定的网络连接访问DashScope服务
3. **地域选择**: 根据使用地区选择合适的base_url
4. **模型版本**: 注意区分不同版本的模型能力差异
5. **计费方式**: 了解不同模型的计费标准和方式

## 最佳实践

### 1. 模型选择策略
```python
# 根据任务类型选择合适的模型
simple_tasks = "qwen-turbo"           # 简单任务
complex_tasks = "qwen-max"            # 复杂任务
visual_tasks = "qwen-vl-max"          # 视觉任务
coding_tasks = "qwen-coder-plus"      # 代码任务
math_tasks = "qwen-math-plus"         # 数学任务
long_context = "qwen-long"            # 长文本任务
```

### 2. 温度设置建议
```python
creative_temp = 0.8     # 创意写作
balanced_temp = 0.5     # 平衡输出
factual_temp = 0.2      # 事实性问答
```

### 3. 中文提示词优化
- 使用清晰的中文指令
- 提供具体的上下文信息
- 善用中文的表达习惯

### 4. 性能优化
- 合理选择模型规模
- 控制上下文长度
- 使用合适的temperature设置
