# 接口文档

## 类：`Model`

### 初始化方法：`__init__(self, interpreter: Interpreter[S], echo: bool = True) -> None`
- **参数说明**：
  - `interpreter`: 一个 `Interpreter` 实例，用于执行模型。
  - `echo`: 一个布尔值，指示是否在执行时回显输出，默认为 `True`。
- **返回值说明**：无返回值。

### 属性：
- `echo`: 布尔值，指示是否回显输出。
- `_interpreter`: `Interpreter` 实例，用于执行模型。
- `_active_blocks`: 字典，存储当前活动的块及其起始索引。
- `token_count`: 整数，记录生成的令牌数量。
- `_parent`: 可选的 `Model` 实例，指向父模型。
- `_parent_id`: 可选的整数，父模型的 ID。
- `_id`: 整数，模型的唯一标识符。
- `_trace_nodes`: 集合，存储跟踪节点。

### 方法：
#### `__add__(self, other: Union[str, Function, ASTNode]) -> Self`
- **参数说明**：
  - `other`: 可以是字符串、函数或 AST 节点。
- **返回值说明**：返回更新后的 `Model` 实例。
- **范围说明**：支持字符串、函数和 AST 节点的组合。

#### `stream(self) -> "ModelStream"`
- **参数说明**：无参数。
- **返回值说明**：返回一个新的 `ModelStream` 实例。
- **范围说明**：用于延迟执行模型。

#### `copy(self) -> Self`
- **参数说明**：无参数。
- **返回值说明**：返回当前模型的深拷贝。
- **范围说明**：用于创建模型的副本。

#### `get(self, key: str, default: Optional[D] = None) -> Union[str, list[str], None, D]`
- **参数说明**：
  - `key`: 字符串，变量的名称。
  - `default`: 可选，未设置时返回的默认值。
- **返回值说明**：返回变量的值或默认值。
- **范围说明**：用于获取模型中的变量值。

#### `set(self, key: str, value: Union[str, list[str]]) -> Self`
- **参数说明**：
  - `key`: 字符串，变量的名称。
  - `value`: 字符串或字符串列表，设置的值。
- **返回值说明**：返回更新后的 `Model` 实例。
- **范围说明**：用于设置模型中的变量值。

#### `remove(self, key: str) -> Self`
- **参数说明**：
  - `key`: 字符串，变量的名称。
- **返回值说明**：返回更新后的 `Model` 实例。
- **范围说明**：用于删除模型中的变量。

#### `log_prob(self, key: str, default: Optional[D] = None) -> Union[float, list[Union[float, None]], None, D]`
- **参数说明**：
  - `key`: 字符串，变量的名称。
  - `default`: 可选，未设置时返回的默认值。
- **返回值说明**：返回变量的对数概率或默认值。
- **范围说明**：用于获取变量的对数概率。

## 类：`ModelStream`

### 初始化方法：`__init__(self, model: Model, grammar: Union["ModelStream", str, ASTNode, Function, None] = None, timeout=5) -> None`
- **参数说明**：
  - `model`: 一个 `Model` 实例。
  - `grammar`: 可选，可以是 `ModelStream`、字符串、AST 节点或函数，默认为 `None`。
  - `timeout`: 整数，超时时间，默认为 5 秒。
- **返回值说明**：无返回值。

### 方法：
#### `__add__(self, grammar: Union[str, ASTNode]) -> Self`
- **参数说明**：
  - `grammar`: 字符串或 AST 节点，用于扩展执行链。
- **返回值说明**：返回更新后的 `ModelStream` 实例。
- **范围说明**：用于在延迟执行链中添加语法。

#### `__iter__(self) -> Iterator[Model]`
- **参数说明**：无参数。
- **返回值说明**：返回一个迭代器，逐步生成 `Model` 实例。
- **范围说明**：用于启动线程执行模型并生成事件。

## 函数：`_gen_id()`
- **参数说明**：无参数。
- **返回值说明**：返回一个唯一的整数 ID。
- **范围说明**：用于生成模型的唯一标识符。

## 调用示例

```python
# 创建一个 Interpreter 实例
interpreter = Interpreter()

# 创建 Model 实例
model = Model(interpreter)

# 设置变量
model = model.set("key", "value")

# 获取变量
value = model.get("key")

# 创建 ModelStream 实例
stream = model.stream()

# 迭代 ModelStream
for event in stream:
    print(event)
```