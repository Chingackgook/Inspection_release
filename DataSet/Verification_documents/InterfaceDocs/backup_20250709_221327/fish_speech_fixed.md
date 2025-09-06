# 接口文档

## 1. 函数接口说明

### 函数名: `load_model`

#### 参数说明:
- `config_name` (str): 配置文件的名称，用于动态加载模型参数。
- `checkpoint_path` (str): 模型检查点的路径，用于加载模型的状态字典。
- `device` (str, optional): 指定设备，默认为 "cuda"。可选值包括 "cuda" 或 "cpu"。

#### 返回值说明:
- 返回加载后的模型实例。

#### 范围说明:
- 该函数用于加载模型的状态字典，并将其设置为评估模式，移动到指定的设备上。

#### 调用示例:
```python
model = load_model(config_name="my_model_config", checkpoint_path="path/to/checkpoint.pth", device="cuda")
```

---

## 2. 类接口说明

### 类名: `GlobalHydra`

#### 初始化信息:
- `GlobalHydra` 是一个单例类，负责管理全局的 Hydra 实例。

#### 属性:
- `instance`: 返回 `GlobalHydra` 的单例实例。

#### 方法:
- `clear()`: 清除当前的全局 Hydra 实例。

---

### 类名: `cfg`

#### 初始化信息:
- `cfg` 是通过 Hydra 加载的配置对象，包含模型的配置参数。

#### 属性:
- `model`: 模型的配置参数。

#### 方法:
- `instantiate()`: 根据配置参数实例化模型。

---

### 类名: `model`

#### 初始化信息:
- `model` 是通过 `instantiate(cfg)` 创建的模型实例。

#### 属性:
- `state_dict`: 模型的状态字典。

#### 方法:
- `load_state_dict(state_dict, strict=False, assign=True)`: 加载状态字典到模型中。
- `eval()`: 将模型设置为评估模式。
- `to(device)`: 将模型移动到指定的设备上。
- `encode(audios, audio_lengths)`: 提取音频特征。
- `decode(indices, feature_lengths)`: 将离散的音频特征转换为连续的音频。

---

## 3. 其他方法接口说明

### 方法名: `encode`

#### 参数说明:
- `audios` (Tensor): 输入的音频数据。
- `audio_lengths` (Tensor): 输入音频的长度。

#### 返回值说明:
- 返回提取的音频特征。

#### 范围说明:
- 该方法用于从输入音频中提取特征。

---

### 方法名: `decode`

#### 参数说明:
- `indices` (Tensor): 离散音频特征的索引。
- `feature_lengths` (Tensor): 特征的长度。

#### 返回值说明:
- 返回转换后的连续音频数据。

#### 范围说明:
- 该方法用于将离散的音频特征转换为连续的音频。

---

## 4. 完整调用示例

```python
# 加载模型
model = load_model(config_name="my_model_config", checkpoint_path="path/to/checkpoint.pth", device="cuda")

# 提取音频特征
indices = model.encode(audios, audio_lengths)[0][0]

# 将离散的音频特征转换为连续的音频
fake_audios, _ = model.decode(indices=indices[None], feature_lengths=feature_lengths)

# 计算生成音频的时间
audio_time = fake_audios.shape[-1] / model.spec_transform.sample_rate

# 日志记录生成的音频信息
logger.info(f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {indices.shape[1]} features, features/second: {indices.shape[1] / audio_time:.2f}")

# 保存音频
fake_audio = fake_audios[0, 0].float().cpu().numpy()
sf.write(output_path, fake_audio, model.spec_transform.sample_rate)
```