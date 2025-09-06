# Ollama本地大模型适配器

这是一个用于与Ollama本地大模型服务交互的适配器，基于Ollama Python客户端实现。

## 功能特性

- ✅ **本地运行**: 在本地机器上运行大语言模型，数据隐私有保障
- ✅ **文本生成**: 支持各种开源大语言模型
- ✅ **图片处理**: 支持多模态模型进行图像分析
- ✅ **音频处理**: 提供音频文件信息处理（有限支持）
- ✅ **错误重试**: 内置重试机制，提高稳定性
- ✅ **历史记录**: 支持对话历史管理
- ✅ **性能统计**: 提供推理时间和token统计
- ✅ **模型管理**: 支持模型列表查看、拉取等管理功能
- ✅ **完全免费**: 无API费用，只需本地计算资源
- ✅ **隐私安全**: 数据不离开本地环境

## Ollama简介

Ollama是一个强大的本地大模型运行工具，它让在本地运行大语言模型变得非常简单。通过Ollama，你可以：

- 在本地CPU或GPU上运行各种开源大模型
- 享受与ChatGPT类似的对话体验
- 保护数据隐私，模型推理完全在本地进行
- 支持自定义模型和微调
- 提供RESTful API接口

## 支持的模型

### 主流大语言模型

#### Meta Llama系列
- **llama3.2** (1B, 3B) - 最新Llama 3.2模型，轻量高效
- **llama3.2:1b** - 1B参数版本，快速响应
- **llama3.2:3b** - 3B参数版本，平衡性能
- **llama3.1** (8B, 70B, 405B) - Llama 3.1系列
- **llama3** (8B, 70B) - Llama 3.0系列
- **llama2** (7B, 13B, 70B) - Llama 2系列
- **llama2-uncensored** - 无审查版本

#### Google Gemma系列
- **gemma2** (2B, 9B, 27B) - Google最新Gemma 2系列
- **gemma** (2B, 7B) - Google Gemma系列

#### Mistral AI系列
- **mistral** (7B) - Mistral 7B基础模型
- **mixtral** (8x7B, 8x22B) - Mixtral混合专家模型
- **mistral-nemo** (12B) - 轻量级Mistral模型

#### 微软系列
- **phi3** (3B, 14B) - 微软Phi-3小型语言模型
- **phi3.5** - Phi-3.5增强版本

#### 阿里巴巴系列
- **qwen2** (0.5B, 1.5B, 7B, 72B) - 阿里巴巴通义千问2.0
- **qwen2.5** (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B) - 通义千问2.5
- **qwen** (4B, 7B, 14B, 72B) - 通义千问1.0系列

#### 其他优秀模型
- **deepseek-coder** (1.3B, 6.7B, 33B) - 专业代码生成模型
- **codellama** (7B, 13B, 34B) - Meta代码专用模型
- **wizardcoder** (15B, 34B) - 代码生成优化模型
- **vicuna** (7B, 13B, 33B) - 基于Llama的对话模型
- **orca-mini** (3B, 7B, 13B, 70B) - 微软Orca系列小型版本

### 多模态模型

#### 视觉理解模型
- **llama3.2-vision** (11B, 90B) - Llama 3.2视觉版本
- **llava** (7B, 13B, 34B) - 大型语言和视觉助手
- **llava-phi3** - 基于Phi-3的视觉模型
- **bakllava** (7B) - 改进的视觉理解模型

#### 专业视觉模型
- **moondream** (1.4B) - 轻量级视觉理解模型
- **obsidian** (3B) - 专业视觉分析模型

### 专业领域模型

#### 代码生成
- **codeqwen** (7B) - 阿里巴巴代码专用模型
- **starcoder2** (3B, 7B, 15B) - BigCode项目代码模型
- **magicoder** (7B) - 代码生成优化模型

#### 数学推理
- **mathstral** (7B) - Mistral数学专用模型
- **deepseek-math** (7B) - 深度求索数学模型

#### 多语言支持
- **aya** (8B, 35B) - 多语言对话模型
- **solar** (10.7B) - 韩语优化模型

## 模型选择建议

| 使用场景 | 推荐模型 | 特点 |
|----------|----------|------|
| 轻量快速 | llama3.2:1b, gemma2:2b | 资源占用小，响应快 |
| 平衡性能 | llama3.2:3b, phi3:3b | 性能与资源平衡 |
| 高质量对话 | llama3.1:8b, qwen2.5:7b | 对话质量高 |
| 代码生成 | deepseek-coder, codellama | 代码专业能力强 |
| 视觉理解 | llama3.2-vision, llava | 多模态处理 |
| 中文优化 | qwen2.5, qwen2 | 中文理解优秀 |
| 大参数模型 | llama3.1:70b, mixtral:8x22b | 最强能力（需要大内存） |
| 数学推理 | mathstral, deepseek-math | 数学专业能力 |

## 安装依赖

### 1. 安装Ollama

#### macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
# 或使用 Homebrew
brew install ollama
```

#### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows
下载并安装 [Ollama Windows版本](https://ollama.com/download)

### 2. 安装Python依赖
```bash
pip install ollama pillow
```

### 3. 启动Ollama服务
```bash
ollama serve
```

### 4. 拉取模型
```bash
# 拉取推荐的基础模型
ollama pull llama3.2
ollama pull llama3.2-vision

# 拉取其他模型（示例）
ollama pull gemma2:2b
ollama pull qwen2.5:3b
ollama pull deepseek-coder
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "ollama_host": "http://localhost:11434",
    "ollama_model": "llama3.2",
    "ollama_model_img": "llama3.2-vision",
    "ollama_model_audio": "llama3.2",
    "ollama_keep_alive": "5m"
}
```

### 配置说明

- **ollama_host**: Ollama服务地址，默认为本地11434端口
- **ollama_model**: 用于文本生成的默认模型
- **ollama_model_img**: 用于图片处理的多模态模型
- **ollama_model_audio**: 用于音频处理的模型（当前作为文本处理）
- **ollama_keep_alive**: 模型在内存中保持活跃的时间
  - `5m` - 5分钟后卸载
  - `30s` - 30秒后卸载
  - `-1` - 永远保持在内存中
  - `0` - 立即卸载

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Ollama_adapter import OllamaAdapter

# 初始化适配器
adapter = OllamaAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下Ollama",
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
    prompt="请分析这些音频文件的基本信息",
    filepath=["path/to/audio.mp3"],
    temperature=0.3,
    max_tokens=1000
)

print(f"音频处理: {response}")
```

### 模型管理

```python
# 获取本地模型列表
models = adapter.list_models()
for model in models:
    print(f"模型: {model['name']}, 大小: {model['size']}")

# 检查模型是否存在
if adapter.check_model_exists("llama3.2"):
    print("模型已存在")
else:
    # 拉取模型
    adapter.pull_model("llama3.2")
```

### 在项目中使用

要在项目的 BaseAI 中使用 Ollama 适配器，需要：

1. 确保Ollama服务正在运行
2. 在 `config.json` 中设置 `"provider": "Ollama"`
3. 确保目标模型已经拉取到本地

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载 Ollama 适配器
ai = BaseAI()
response = ai.generate_text("请用中文介绍一下人工智能的发展历程")
```

## 技术细节

### Ollama Python客户端
本适配器基于官方Ollama Python客户端：
- 使用 `ollama.Client` 进行连接管理
- 支持 `chat` API进行对话
- 支持流式和非流式响应
- 自动处理模型加载和卸载

### 多模态处理
对于图片处理：
- 使用base64编码传输图片
- 支持llama3.2-vision等多模态模型
- 自动检测并使用合适的模型

### 性能优化
- `keep_alive` 参数控制模型在内存中的保持时间
- 支持GPU加速（如果可用）
- 自动重试机制提高稳定性

### 对话历史格式
适配器处理对话历史格式：
- 支持 "user", "assistant", "system" 角色
- 自动转换为Ollama兼容格式
- 保持对话上下文连贯性

## 性能调优

### 1. 模型选择策略
```python
# 根据可用资源选择模型
available_memory_gb = 16

if available_memory_gb >= 64:
    model = "llama3.1:70b"  # 高端配置
elif available_memory_gb >= 32:
    model = "llama3.1:8b"   # 中端配置
elif available_memory_gb >= 16:
    model = "llama3.2:3b"   # 标准配置
else:
    model = "llama3.2:1b"   # 低端配置
```

### 2. Keep-Alive设置
```python
# 频繁使用：保持在内存中
"ollama_keep_alive": "-1"

# 偶尔使用：适度保持
"ollama_keep_alive": "10m"

# 内存紧张：快速释放
"ollama_keep_alive": "30s"
```

### 3. GPU加速
```bash
# 检查GPU可用性
nvidia-smi

# 启动Ollama时自动使用GPU
ollama serve
```
