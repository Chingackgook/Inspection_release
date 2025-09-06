# 百度文心一言适配器

## 支持的模型

### ERNIE文本模型

#### 主力商用模型
- **ERNIE-4.0-8K**: 文心大模型4.0，百度最新一代知识增强大语言模型
- **ERNIE-3.5-8K**: 文心大模型3.5，中文能力突出的大语言模型
- **ERNIE-3.5-8K-0205**: ERNIE-3.5的稳定版本
- **ERNIE-Speed-128K**: 高速推理版本，支持128K上下文
- **ERNIE-Speed-8K**: 高速推理版本，平衡性能与速度

#### 专业模型
- **ERNIE-Lite-8K**: 轻量化版本，快速响应
- **ERNIE-Tiny-8K**: 超轻量版本，成本最优
- **ERNIE-Character-8K**: 角色扮演专用模型
- **ERNIE-Functions-8K**: 函数调用优化模型

#### 开源模型
- **Llama-2-7B-Chat**: Meta Llama2 7B对话版本
- **Llama-2-13B-Chat**: Meta Llama2 13B对话版本
- **Llama-2-70B-Chat**: Meta Llama2 70B对话版本
- **Qianfan-Chinese-Llama-2-7B**: 千帆中文Llama2 7B版本
- **ChatGLM2-6B-32K**: 清华ChatGLM2 6B长文本版本
- **AquilaChat-7B**: 智源悟道AquilaChat 7B模型

### 多模态模型
- **ERNIE-VilG-v2**: 文心一格2.0，文生图模型
- **ERNIE-Vision**: 图像理解模型（实验性）

### 嵌入模型
- **Embedding-V1**: 文本向量化模型
- **bge-large-zh**: 中文文本嵌入模型
- **bge-large-en**: 英文文本嵌入模型

## 模型选择建议

| 使用场景 | 推荐模型 | 特点 |
|----------|----------|------|
| 高质量对话 | ERNIE-4.0-8K | 最强能力，最新技术 |
| 通用任务 | ERNIE-3.5-8K | 平衡性能与成本 |
| 快速响应 | ERNIE-Speed-8K | 高速推理，低延迟 |
| 长文档处理 | ERNIE-Speed-128K | 支持128K上下文 |
| 成本优化 | ERNIE-Lite-8K | 轻量化，成本低 |
| 角色扮演 | ERNIE-Character-8K | 角色扮演专用 |
| 函数调用 | ERNIE-Functions-8K | 工具调用优化 |
| 开源部署 | Qianfan-Chinese-Llama-2-7B | 中文优化开源模型 |

## 安装依赖

```bash
pip install qianfan pillow
```

## 配置设置

在 `Inspection/config.json` 文件中添加以下配置：

```json
{
    "baidu_access_key": "your_baidu_access_key_here",
    "baidu_secret_key": "your_baidu_secret_key_here",
    "baidu_model": "ERNIE-3.5-8K",
    "baidu_model_img": "ERNIE-VilG-v2",
    "baidu_model_audio": "ERNIE-3.5-8K"
}
```

### 获取百度智能云密钥

1. 访问 [百度智能云控制台](https://console.bce.baidu.com/)
2. 进入 [千帆大模型平台](https://console.bce.baidu.com/qianfan/overview)
3. 在"安全认证"页面获取Access Key和Secret Key
4. 或者在"应用接入"页面创建应用获取API Key和Secret Key

### 认证方式选择

百度千帆支持两种认证方式：

#### 方式一：Access Key + Secret Key（推荐）
- 支持全部功能：对话、训练、模型管理等
- 适用于企业级应用

#### 方式二：API Key + Secret Key
- 仅支持模型推理功能：对话、嵌入等
- 适用于简单的API调用

## 使用方法

### 基本文本生成

```python
from Inspection.ai.ai_adapters.Baidu_adapter import BaiduAdapter

# 初始化适配器
adapter = BaiduAdapter()

# 生成文本
history = []
response, token_usage = adapter.generate_text(
    history=history,
    promote="你好，请介绍一下文心一言",
    temperature=0.3,
    max_tokens=1000
)

print(f"回复: {response}")
print(f"Token使用量: {token_usage}")
```

### 图片处理

```python
# 分析图片（基于文件信息）
response = adapter.generate_image(
    history=history,
    prompt="请分析这些图片的基本信息和可能内容",
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

要在项目的 BaseAI 中使用百度适配器，需要：

1. 在 `config.json` 中设置 `"provider": "Baidu"`
2. 确保已配置正确的百度智能云密钥

```python
from Inspection.ai.base_ai import BaseAI

# BaseAI 会自动加载百度适配器
ai = BaseAI()
response = ai.generate_text("请用中文介绍一下人工智能的发展历程")
```

## 技术细节

### 千帆SDK集成
本适配器基于百度官方千帆SDK实现：
- 使用 `qianfan.ChatCompletion` 进行对话
- 支持流式和非流式响应
- 自动处理认证和token管理

### 温度参数处理
千帆API的温度参数范围是0.0-1.0：
- 如果传入的 temperature > 1，会自动除以100进行转换
- 例如：传入30会转换为0.3

### 对话历史格式
适配器自动处理对话历史格式转换：
- 支持 "user", "assistant" 角色
- "system" 角色会转换为特殊格式的 "user" 消息
- 保持对话的连贯性和上下文

### 图片处理说明
当前版本的图片处理基于文件信息分析：
- 提取图片基本信息（尺寸、格式、大小）
- 生成描述性文本进行分析
- 未来版本将集成真正的视觉理解能力
