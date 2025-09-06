# 接口文档

## 类: `TextToSpeech`
### 描述
实现了一个基于深度学习的多阶段文本到语音（TTS）合成系统，结合了自回归模型、扩散模型和语音质量评估模型。

### 属性
- `models_dir`: 模型权重存储的目录。
- `autoregressive_batch_size`: 每个批次生成的样本数量。
- `enable_redaction`: 是否启用文本遮蔽功能。
- `device`: 运行模型时使用的设备。
- `tokenizer`: 用于文本编码的分词器。
- `half`: 是否使用半精度浮点数。
- `autoregressive`: 自回归模型。
- `diffusion`: 扩散模型。
- `clvp`: CLVP模型。
- `cvvp`: CVVP模型（仅在需要时加载）。
- `vocoder`: Vocoder模型。
- `stft`: STFT模型（仅在需要时加载）。
- `rlg_auto`: 随机潜在生成器（自回归）。
- `rlg_diffusion`: 随机潜在生成器（扩散）。

### 方法
#### `__init__(self, autoregressive_batch_size=None, models_dir=MODELS_DIR, enable_redaction=True, kv_cache=False, use_deepspeed=False, half=False, device=None, tokenizer_vocab_file=None, tokenizer_basic=False)`
- **参数**:
  - `autoregressive_batch_size`: 每个批次生成的样本数量，默认为None。
  - `models_dir`: 模型权重存储的目录，默认为`MODELS_DIR`。
  - `enable_redaction`: 是否启用文本遮蔽功能，默认为True。
  - `kv_cache`: 是否启用键值缓存，默认为False。
  - `use_deepspeed`: 是否使用DeepSpeed，默认为False。
  - `half`: 是否使用半精度浮点数，默认为False。
  - `device`: 运行模型时使用的设备，默认为None。
  - `tokenizer_vocab_file`: 分词器的词汇文件，默认为None。
  - `tokenizer_basic`: 是否使用基本清理器，默认为False。
- **返回值**: 无
- **作用**: 初始化`TextToSpeech`类的实例，加载模型和设置参数。

#### `temporary_cuda(self, model)`
- **参数**:
  - `model`: 要临时移动到CUDA设备的模型。
- **返回值**: 生成的模型在上下文管理器中。
- **作用**: 将模型临时移动到CUDA设备，并在上下文结束时将其移回CPU。

#### `load_cvvp(self)`
- **参数**: 无
- **返回值**: 无
- **作用**: 加载CVVP模型。

#### `get_conditioning_latents(self, voice_samples, return_mels=False)`
- **参数**:
  - `voice_samples`: 包含2个或更多约10秒参考片段的列表，应该是包含22.05kHz波形数据的torch张量。
  - `return_mels`: 是否返回梅尔频谱，默认为False。
- **返回值**: 
  - 如果`return_mels`为True，返回一个元组`(autoregressive_conditioning_latent, diffusion_conditioning_latent, auto_conds, diffusion_conds)`；否则返回`(autoregressive_conditioning_latent, diffusion_conditioning_latent)`。
- **作用**: 将一个或多个语音样本转换为自回归和扩散模型的条件潜在表示。

#### `get_random_conditioning_latents(self)`
- **参数**: 无
- **返回值**: 返回自回归和扩散模型的随机条件潜在表示。
- **作用**: 获取随机条件潜在表示，用于生成语音。

#### `tts_with_preset(self, text, preset='fast', **kwargs)`
- **参数**:
  - `text`: 要生成语音的文本。
  - `preset`: 生成参数的预设，默认为'fast'。
  - `**kwargs`: 其他可选参数，用于覆盖预设设置。
- **返回值**: 生成的音频片段，作为torch张量。
- **作用**: 使用预设的生成参数调用TTS。

#### `tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None, return_deterministic_state=False, num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=500, cvvp_amount=.0, diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0, **hf_generate_kwargs)`
- **参数**:
  - `text`: 要生成语音的文本。
  - `voice_samples`: 包含2个或更多约10秒参考片段的列表，应该是包含22.05kHz波形数据的torch张量。
  - `conditioning_latents`: 自回归和扩散模型的条件潜在表示元组。
  - `k`: 返回的音频片段数量，默认为1。
  - `verbose`: 是否打印生成进度，默认为True。
  - `use_deterministic_seed`: 用于生成确定性结果的种子，默认为None。
  - `return_deterministic_state`: 是否返回确定性状态，默认为False。
  - `num_autoregressive_samples`: 自回归模型生成的样本数量，默认为512。
  - `temperature`: 自回归模型的温度，默认为0.8。
  - `length_penalty`: 自回归解码器的长度惩罚，默认为1。
  - `repetition_penalty`: 自回归解码器的重复惩罚，默认为2.0。
  - `top_p`: 用于核采样的P值，默认为0.8。
  - `max_mel_tokens`: 输出长度限制，默认为500。
  - `cvvp_amount`: CVVP模型的影响程度，默认为0.0。
  - `diffusion_iterations`: 扩散步骤的数量，默认为100。
  - `cond_free`: 是否执行无条件扩散，默认为True。
  - `cond_free_k`: 无条件信号与有条件信号的平衡参数，默认为2。
  - `diffusion_temperature`: 扩散模型中噪声的方差，默认为1.0。
  - `**hf_generate_kwargs`: 额外的关键字参数，转发给Hugging Face生成API。
- **返回值**: 生成的音频片段，作为torch张量。
- **作用**: 生成给定文本的音频片段，使用指定的参考语音。

#### `deterministic_state(self, seed=None)`
- **参数**:
  - `seed`: 用于设置随机种子的值，默认为None。
- **返回值**: 返回设置的种子值。
- **作用**: 设置随机种子以确保结果可重现。

#### `potentially_redact(self, clip, text)`
- **参数**:
  - `clip`: 要处理的音频片段。
  - `text`: 原始文本，用于遮蔽处理。
- **返回值**: 处理后的音频片段。
- **作用**: 如果启用遮蔽功能，则对音频片段进行遮蔽处理。