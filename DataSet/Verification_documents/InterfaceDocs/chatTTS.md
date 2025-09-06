# 接口文档

## 类: `Chat`
`Chat` 类用于实现聊天功能，提供模型加载、推理、音频生成等功能。

### 初始化方法: `__init__`
```python
def __init__(self, logger=logging.getLogger(__name__)):
```
#### 参数说明:
- `logger`: (可选) 用于记录日志的 Logger 对象，默认为当前模块的 Logger。

#### 属性:
- `logger`: 记录日志的 Logger 对象。
- `config`: 配置对象，包含模型路径等配置信息。
- `normalizer`: 文本归一化器，用于处理输入文本。
- `sha256_map`: 存储模型文件的 SHA256 校验和的字典。
- `context`: GPT 上下文对象。

#### 返回值:
无

#### 作用简述:
初始化 `Chat` 类的实例，设置日志记录器和加载必要的配置与资源。

---

### 方法: `has_loaded`
```python
def has_loaded(self, use_decoder=False):
```
#### 参数说明:
- `use_decoder`: (可选) 布尔值，指示是否检查解码器模块的加载状态，默认为 `False`。

#### 返回值:
- `bool`: 如果所有必要模块已加载，返回 `True`；否则返回 `False`。

#### 作用简述:
检查所需模块是否已成功加载。

---

### 方法: `download_models`
```python
def download_models(
    self,
    source: Literal["huggingface", "local", "custom"] = "local",
    force_redownload=False,
    custom_path: Optional[torch.serialization.FILE_LIKE] = None,
) -> Optional[str]:
```
#### 参数说明:
- `source`: (可选) 字符串，指定模型下载源，默认为 `"local"`。可选值包括 `"huggingface"`、`"local"` 和 `"custom"`。
- `force_redownload`: (可选) 布尔值，指示是否强制重新下载模型，默认为 `False`。
- `custom_path`: (可选) 自定义路径，用于指定模型文件的本地路径。

#### 返回值:
- `Optional[str]`: 返回下载的模型路径，如果下载失败则返回 `None`。

#### 作用简述:
根据指定的源下载模型文件。

---

### 方法: `load`
```python
def load(
    self,
    source: Literal["huggingface", "local", "custom"] = "local",
    force_redownload=False,
    compile: bool = False,
    custom_path: Optional[torch.serialization.FILE_LIKE] = None,
    device: Optional[torch.device] = None,
    coef: Optional[torch.Tensor] = None,
    use_flash_attn=False,
    use_vllm=False,
    experimental: bool = False,
) -> bool:
```
#### 参数说明:
- `source`: (可选) 字符串，指定模型加载源，默认为 `"local"`。
- `force_redownload`: (可选) 布尔值，指示是否强制重新下载模型，默认为 `False`。
- `compile`: (可选) 布尔值，指示是否编译模型，默认为 `False`。
- `custom_path`: (可选) 自定义路径，用于指定模型文件的本地路径。
- `device`: (可选) 指定模型运行的设备。
- `coef`: (可选) 用于模型的系数。
- `use_flash_attn`: (可选) 布尔值，指示是否使用 Flash Attention，默认为 `False`。
- `use_vllm`: (可选) 布尔值，指示是否使用 VLLM，默认为 `False`。
- `experimental`: (可选) 布尔值，指示是否使用实验性功能，默认为 `False`。

#### 返回值:
- `bool`: 如果模型加载成功返回 `True`，否则返回 `False`。

#### 作用简述:
加载指定源的模型文件。

---

### 方法: `unload`
```python
def unload(self):
```
#### 参数说明:
无

#### 返回值:
无

#### 作用简述:
卸载已加载的模型和资源，释放内存。

---

### 方法: `sample_random_speaker`
```python
def sample_random_speaker(self) -> str:
```
#### 参数说明:
无

#### 返回值:
- `str`: 随机选择的说话者的标识符。

#### 作用简述:
从可用的说话者中随机选择一个。

---

### 方法: `sample_audio_speaker`
```python
def sample_audio_speaker(self, wav: Union[np.ndarray, torch.Tensor]) -> str:
```
#### 参数说明:
- `wav`: 输入的音频数据，可以是 NumPy 数组或 PyTorch 张量。

#### 返回值:
- `str`: 编码后的说话者标识符。

#### 作用简述:
根据输入的音频数据编码并返回相应的说话者标识符。

---

### 方法: `infer`
```python
def infer(
    self,
    text,
    stream=False,
    lang=None,
    skip_refine_text=False,
    refine_text_only=False,
    use_decoder=True,
    do_text_normalization=True,
    do_homophone_replacement=True,
    split_text=True,
    max_split_batch=4,
    params_refine_text=RefineTextParams(),
    params_infer_code=InferCodeParams(),
):
```
#### 参数说明:
- `text`: 输入的文本，可以是字符串或字符串列表。
- `stream`: (可选) 布尔值，指示是否以流式方式返回结果，默认为 `False`。
- `lang`: (可选) 指定语言。
- `skip_refine_text`: (可选) 布尔值，指示是否跳过文本精炼，默认为 `False`。
- `refine_text_only`: (可选) 布尔值，指示是否仅进行文本精炼，默认为 `False`。
- `use_decoder`: (可选) 布尔值，指示是否使用解码器，默认为 `True`。
- `do_text_normalization`: (可选) 布尔值，指示是否进行文本归一化，默认为 `True`。
- `do_homophone_replacement`: (可选) 布尔值，指示是否进行同音词替换，默认为 `True`。
- `split_text`: (可选) 布尔值，指示是否将文本拆分，默认为 `True`。
- `max_split_batch`: (可选) 整数，指定最大拆分批次，默认为 `4`。
- `params_refine_text`: (可选) 文本精炼参数，默认为 `RefineTextParams()`。
- `params_infer_code`: (可选) 推理代码参数，默认为 `InferCodeParams()`。

#### 返回值:
- `Union[List[np.ndarray], np.ndarray]`: 返回生成的音频数据，可能是音频数组的列表或单个音频数组。

#### 作用简述:
根据输入文本生成音频数据。

---

### 方法: `interrupt`
```python
def interrupt(self):
```
#### 参数说明:
无

#### 返回值:
无

#### 作用简述:
中断当前的推理过程。