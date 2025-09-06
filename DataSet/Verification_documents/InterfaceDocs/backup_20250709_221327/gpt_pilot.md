# 接口文档：`BaseLLMClient`

## 类：`BaseLLMClient`

### 初始化方法：`__init__`
```python
def __init__(
    self,
    config: LLMConfig,
    *,
    stream_handler: Optional[Callable] = None,
    error_handler: Optional[Callable] = None,
)
```

#### 参数说明：
- `config` (`LLMConfig`): 客户端的配置。
- `stream_handler` (`Optional[Callable]`): 可选的流式响应处理器，接收单个参数（响应内容字符串）。
- `error_handler` (`Optional[Callable]`): 可选的错误处理器，处理错误情况。

#### 返回值说明：
- 无返回值。

---

### 属性：`provider`
- 类型：`LLMProvider`
- 描述：表示当前使用的语言模型提供者。

---

### 方法：`_init_client`
```python
def _init_client(self)
```

#### 返回值说明：
- 无返回值。
- 抛出`NotImplementedError`，表示该方法需要在子类中实现。

---

### 方法：`_make_request`
```python
async def _make_request(
    self,
    convo: Convo,
    temperature: Optional[float] = None,
    json_mode: bool = False,
) -> tuple[str, int, int]
```

#### 参数说明：
- `convo` (`Convo`): 要发送到语言模型的对话。
- `temperature` (`Optional[float]`): 温度设置影响响应的随机性（默认使用配置中的温度）。
- `json_mode` (`bool`): 如果为`True`，则预计响应为 JSON 格式。

#### 返回值说明：
- 返回 `tuple[str, int, int]`，包括：
    1. 完整的响应内容（字符串）。
    2. 输入 tokens 的数量（整数）。
    3. 输出 tokens 的数量（整数）。

#### 范围说明：
- 抛出`NotImplementedError`，表示该方法需要在子类中实现。

---

### 方法：`_adapt_messages`
```python
async def _adapt_messages(self, convo: Convo) -> list[dict[str, str]]
```

#### 参数说明：
- `convo` (`Convo`): 要适应的对话。

#### 返回值说明：
- 返回 `list[dict[str, str]]`，适应后的对话消息。每个字典包含`role`（角色）和`content`（内容）。

---

### 方法：`__call__`
```python
async def __call__(
    self,
    convo: Convo,
    *,
    temperature: Optional[float] = None,
    parser: Optional[Callable] = None,
    max_retries: int = 3,
    json_mode: bool = False,
) -> Tuple[Any, LLMRequestLog]
```

#### 参数说明：
- `convo` (`Convo`): 要发送到 LLM 的对话。
- `temperature` (`Optional[float]`): 温度设置。
- `parser` (`Optional[Callable]`): 可选的响应解析器。
- `max_retries` (`int`): 最大重试次数（默认为 3）。
- `json_mode` (`bool`): 如果为`True`，则预计响应为 JSON 格式。

#### 返回值说明：
- 返回 `Tuple[Any, LLMRequestLog]`，包括解析后的响应和请求日志条目。

---

### 方法：`api_check`
```python
async def api_check(self) -> bool
```

#### 返回值说明：
- 返回 `bool`，表示 API 检查是否成功。

---

### 静态方法：`for_provider`
```python
@staticmethod
def for_provider(provider: LLMProvider) -> type["BaseLLMClient"]
```

#### 参数说明：
- `provider` (`LLMProvider`): 要为其返回客户端的提供者。

#### 返回值说明：
- 返回指定提供者的客户端类类型。

#### 范围说明：
- 如果提供者不支持，则抛出`ValueError`。

---

### 方法：`rate_limit_sleep`
```python
def rate_limit_sleep(self, err: Exception) -> Optional[datetime.timedelta]
```

#### 参数说明：
- `err` (`Exception`): 由 LLM 客户端抛出的 RateLimitError。

#### 返回值说明：
- 返回 `Optional[datetime.timedelta]`，表示需要等待多久再尝试请求。如果没有限制头部，返回`None`。

#### 范围说明：
- 抛出`NotImplementedError`，表示该方法需要在子类中实现。

---

## 调用示例
```python
import json

async def stream_handler(content: str):
    print(content)

def parser(content: str) -> dict:
    return json.loads(content)

client_class = BaseLLMClient.for_provider(LLMProvider.OPENAI)
config = LLMConfig(model="gpt-3.5-turbo", temperature=0.7)
client = client_class(config, stream_handler=stream_handler)

convo = Convo()
convo.user("Hello, how are you?")
response, request_log = await client(convo, parser=parser)
print(response)
```

以上是 `BaseLLMClient` 类及其方法和函数的接口文档，包括参数、返回值及调用示例。