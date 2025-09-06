# 接口文档

## AutoTokenizer

### 类说明
`AutoTokenizer` 是一个通用的分词器类，通过 `from_pretrained` 方法实例化为库中的某个分词器类。该类不能直接通过 `__init__()` 实例化。

### 方法

#### `__init__()`
- **参数**: 无
- **返回值**: 无
- **范围**: 私有
- **作用**: 抛出错误，提示 `AutoTokenizer` 只能通过 `from_pretrained` 方法实例化。

#### `from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)`
- **参数**:
  - `pretrained_model_name_or_path` (`str` 或 `os.PathLike`): 预训练模型的名称或路径。
  - `inputs` (可选): 额外的位置参数，将传递给分词器的 `__init__()` 方法。
  - `config` (`PretrainedConfig`, 可选): 用于确定要实例化的分词器类的配置对象。
  - `cache_dir` (`str` 或 `os.PathLike`, 可选): 下载的预训练模型配置的缓存目录。
  - `force_download` (`bool`, 可选, 默认 `False`): 是否强制重新下载模型权重和配置文件。
  - `resume_download` (`bool`, 可选, 默认 `False`): 是否删除不完整的文件并尝试恢复下载。
  - `proxies` (`Dict[str, str]`, 可选): 用于请求的代理服务器字典。
  - `revision` (`str`, 可选, 默认 `"main"`): 使用的特定模型版本。
  - `subfolder` (`str`, 可选): 如果相关文件位于模型库的子文件夹中，请在此处指定。
  - `use_fast` (`bool`, 可选, 默认 `True`): 是否使用快速的 Rust 基础分词器。
  - `tokenizer_type` (`str`, 可选): 要加载的分词器类型。
  - `trust_remote_code` (`bool`, 可选, 默认 `False`): 是否允许在本地执行 Hub 上自定义模型的代码。
  - `kwargs` (可选): 额外的关键字参数，将传递给分词器的 `__init__()` 方法。
- **返回值**: 返回实例化的分词器类。
- **范围**: 公有
- **作用**: 从预训练模型的词汇中实例化库中的某个分词器类。

#### `register(config_class, slow_tokenizer_class=None, fast_tokenizer_class=None, exist_ok=False)`
- **参数**:
  - `config_class` (`PretrainedConfig`): 要注册的模型对应的配置。
  - `slow_tokenizer_class` (`PretrainedTokenizer`, 可选): 要注册的慢分词器。
  - `fast_tokenizer_class` (`PretrainedTokenizerFast`, 可选): 要注册的快分词器。
  - `exist_ok` (`bool`, 可选): 如果为 `True`，则允许覆盖已存在的注册。
- **返回值**: 无
- **范围**: 公有
- **作用**: 在映射中注册新的分词器。

---

## AutoModel

### 类说明
`AutoModel` 提供统一接口 `from_pretrained()` 动态加载各类预训练模型架构，根据配置自动映射到对应的模型类。

### 方法

#### `__init__()`
- **参数**: 无
- **返回值**: 无
- **范围**: 私有
- **作用**: 抛出错误，提示 `AutoModel` 只能通过 `from_pretrained` 或 `from_config` 方法实例化。

#### `from_config(config, **kwargs)`
- **参数**:
  - `config` (`PretrainedConfig`): 用于加载模型的配置对象。
  - `trust_remote_code` (`bool`, 可选): 是否信任远程代码。
  - `kwargs` (可选): 额外的关键字参数。
- **返回值**: 返回实例化的模型类。
- **范围**: 公有
- **作用**: 从配置对象实例化模型。

#### `from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)`
- **参数**:
  - `pretrained_model_name_or_path` (`str` 或 `os.PathLike`): 预训练模型的名称或路径。
  - `model_args` (可选): 额外的位置参数，将传递给模型的 `__init__()` 方法。
  - `kwargs` (可选): 额外的关键字参数。
- **返回值**: 返回实例化的模型类。
- **范围**: 公有
- **作用**: 从预训练模型加载并实例化模型。

#### `register(config_class, model_class, exist_ok=False)`
- **参数**:
  - `config_class` (`PretrainedConfig`): 要注册的模型对应的配置。
  - `model_class` (`PreTrainedModel`): 要注册的模型。
  - `exist_ok` (`bool`, 可选): 如果为 `True`，则允许覆盖已存在的注册。
- **返回值**: 无
- **范围**: 公有
- **作用**: 在映射中注册新的模型。