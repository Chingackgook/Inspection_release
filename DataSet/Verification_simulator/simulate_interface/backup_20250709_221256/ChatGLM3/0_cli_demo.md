为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要识别出每个函数的调用位置，并将其替换为相应的 `exe.run` 调用。以下是一个方案，详细说明了如何进行替换：

### 替换方案

1. **替换 `AutoTokenizer.from_pretrained` 调用**:
   - 原代码:
     ```python
     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
     ```
   - 替换为:
     ```python
     tokenizer = exe.run("from_pretrained", pretrained_model_name_or_path=TOKENIZER_PATH, trust_remote_code=True)
     ```

2. **替换 `AutoModel.from_pretrained` 调用**:
   - 原代码:
     ```python
     model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
     ```
   - 替换为:
     ```python
     model = exe.run("from_pretrained", pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
     ```

3. **替换 `AutoModel.from_config` 调用** (如果在其他部分有使用):
   - 原代码:
     ```python
     model = AutoModel.from_config(config)
     ```
   - 替换为:
     ```python
     model = exe.run("from_config", config=config)
     ```

4. **替换 `AutoTokenizer.register` 调用** (如果在其他部分有使用):
   - 原代码:
     ```python
     AutoTokenizer.register(config_class, slow_tokenizer_class, fast_tokenizer_class)
     ```
   - 替换为:
     ```python
     exe.run("register", config_class=config_class, slow_tokenizer_class=slow_tokenizer_class, fast_tokenizer_class=fast_tokenizer_class)
     ```

5. **替换 `AutoModel.register` 调用** (如果在其他部分有使用):
   - 原代码:
     ```python
     AutoModel.register(config_class, model_class)
     ```
   - 替换为:
     ```python
     exe.run("register", config_class=config_class, model_class=model_class)
     ```

### 注意事项
- 确保在替换时，所有的参数都正确传递给 `exe.run`，并且保持原有的逻辑不变。
- 如果有其他地方使用了这些函数，也需要进行相应的替换。
- 在替换后，进行充分的测试以确保功能正常，特别是涉及到模型加载和初始化的部分。

通过以上方案，可以将源代码中的关键函数调用替换为 `exe.run` 的形式，从而实现对函数调用的封装和管理。
为了使原代码能够在没有参数的情况下通过 `eval` 函数直接运行，我们需要对代码进行一些修改，以便模拟用户输入和运行时行为。以下是一个方案，详细说明了如何进行这些修改：

### 替换方案

1. **模拟用户输入**:
   - 在原代码中，用户通过 `input()` 函数输入查询内容。为了模拟这一行为，可以在代码中定义一个列表，包含预设的用户输入。然后在主循环中，使用这个列表的元素来替代 `input()` 的调用。
   - 例如，可以定义一个 `user_inputs` 列表，包含多个查询字符串，并使用一个索引来遍历这些输入。

2. **控制程序流**:
   - 由于原代码是一个无限循环，使用 `input()` 进行交互，我们需要在代码中添加一个条件来控制循环的结束。可以在 `user_inputs` 列表中添加一个特定的输入（如 `"stop"`）来模拟用户终止程序的行为。

3. **初始化参数**:
   - 在代码的开头，定义所有需要的参数，例如 `MODEL_PATH` 和 `TOKENIZER_PATH`，并为它们赋予默认值。这些参数可以直接在代码中定义，而不是依赖于环境变量。

4. **移除交互式元素**:
   - 移除所有与用户交互相关的代码，例如 `clear` 命令的处理。可以将这些功能注释掉或删除，以简化代码。

5. **模拟输出**:
   - 为了能够看到程序的输出，可以在每次生成响应后，将其打印到控制台。可以在主循环中添加打印语句，以便在模拟用户输入时查看模型的响应。

### 示例结构

以下是修改后的代码结构示例（不包含具体代码）：

```python
# 初始化参数
MODEL_PATH = './chatglm3-6b'
TOKENIZER_PATH = MODEL_PATH

# 模拟用户输入
user_inputs = [
    "你好",
    "你是谁？",
    "stop"  # 用于终止程序
]

# 初始化模型和分词器
tokenizer = exe.run("from_pretrained", pretrained_model_name_or_path=TOKENIZER_PATH, trust_remote_code=True)
model = exe.run("from_pretrained", pretrained_model_name_or_path=MODEL_PATH, trust_remote_code=True, device_map="auto").eval()

# 主循环
for query in user_inputs:
    if query.strip() == "stop":
        break
    print("\nChatGLM：", end="")
    # 模拟模型响应
    for response, history, past_key_values in model.stream_chat(tokenizer, query, history=history, top_p=1, temperature=0.01, past_key_values=past_key_values, return_past_key_values=True):
        print(response, end="", flush=True)
    print("")
```

### 总结

通过以上方案，我们可以在不改变原有逻辑的情况下，使代码能够在没有参数的情况下通过 `eval` 函数直接运行。主要通过模拟用户输入、控制程序流和初始化参数来实现这一目标。这样，代码在执行时将自动使用预设的输入进行测试，而无需用户交互。