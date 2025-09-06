# 接口文档

## 函数：`evaluate`

### 初始化信息
```python
evaluate(instruction: str, input: Optional[str] = None, temperature: float = 0.1, top_p: float = 0.75, top_k: int = 40, num_beams: int = 4, max_new_tokens: int = 128, stream_output: bool = False, **kwargs)
```
- **参数**:
  - `instruction` (str): 用户输入的指令。
  - `input` (Optional[str]): 可选的输入参数，默认为 `None`。
  - `temperature` (float): 生成文本的温度，默认为 `0.1`。
  - `top_p` (float): 生成文本的top-p采样值，默认为 `0.75`。
  - `top_k` (int): 生成文本的top-k采样值，默认为 `40`。
  - `num_beams` (int): 生成文本的束搜索数量，默认为 `4`。
  - `max_new_tokens` (int): 生成的最大新令牌数，默认为 `128`。
  - `stream_output` (bool): 是否启用流式输出，默认为 `False`。
  - `**kwargs`: 其他可选参数。
- **返回值**:
  - (Generator): 生成的响应字符串的生成器。

### 调用示例
```python
response_generator = evaluate("Tell me about alpacas.", stream_output=True)
for response in response_generator:
    print(response)
```
以上是根据提供的代码生成的接口文档，涵盖了类和函数的初始化信息、参数说明、返回值说明以及调用示例。