# 接口文档

## 1. 类：LlamaTokenizerFast

### 初始化信息
- **构造函数**: `from_pretrained(pretrained_model_name_or_path)`
- **参数**:
  - `pretrained_model_name_or_path`: 预训练模型的路径或名称。

### 属性
- `pad_token`: 填充标记，通常设置为结束标记（`eos_token`）。
- `eos_token`: 结束标记。

### 方法
- `__call__(self, text, return_tensors=None, padding=None, max_length=None, truncation=None)`: 
  - **参数**:
    - `text`: 输入文本。
    - `return_tensors`: 返回的张量类型（如"pt"表示PyTorch）。
    - `padding`: 是否进行填充。
    - `max_length`: 最大长度。
    - `truncation`: 是否截断超出最大长度的文本。
  - **返回值**: 返回编码后的张量。

## 2. 类：LlamaForCausalLM

### 初始化信息
- **构造函数**: `from_pretrained(pretrained_model_name_or_path, device_map=None, torch_dtype=None)`
- **参数**:
  - `pretrained_model_name_or_path`: 预训练模型的路径或名称。
  - `device_map`: 设备映射（如"cpu"）。
  - `torch_dtype`: 张量的数据类型（如`torch.float32`）。

### 方法
- `generate(self, **kwargs)`:
  - **参数**:
    - `**kwargs`: 生成配置参数（如输入张量、最大新令牌数等）。
  - **返回值**: 生成的输出张量。

## 3. 类：PeftModel

### 初始化信息
- **构造函数**: `from_pretrained(base_model, peft_model_name_or_path)`
- **参数**:
  - `base_model`: 基础模型实例。
  - `peft_model_name_or_path`: LoRA适配器的路径或名称。

### 方法
- `eval(self)`:
  - **参数**: 无。
  - **返回值**: None（将模型设置为评估模式）。

## 4. 函数：load_local_model

### 接口说明
- **函数名**: `load_local_model`
- **参数**: 无
- **返回值**: `(model, tokenizer)` - 返回加载的模型和分词器。
- **范围说明**: 加载本地模型和分词器，适用于CPU模式。

### 调用示例
```python
model, tokenizer = load_local_model()
```

## 5. 函数：generate_answer

### 接口说明
- **函数名**: `generate_answer`
- **参数**:
  - `model`: 加载的模型实例。
  - `tokenizer`: 加载的分词器实例。
  - `prompt`: 输入的提示字符串。
- **返回值**: `str` - 生成的答案字符串。
- **范围说明**: 根据输入提示生成模型的回复，处理异常并返回错误信息。

### 调用示例
```python
prompt = "Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.\nInput: Glaxo's ViiV Healthcare Signs China Manufacturing Deal With Desano\nAnswer: "
result = generate_answer(model, tokenizer, prompt)
print(result)
```

## 6. 函数：run_demo_samples

### 接口说明
- **函数名**: `run_demo_samples`
- **参数**:
  - `model`: 加载的模型实例。
  - `tokenizer`: 加载的分词器实例。
- **返回值**: None
- **范围说明**: 运行示例数据，构建提示并调用生成接口，格式化输出结果。

### 调用示例
```python
run_demo_samples(model, tokenizer)
``` 

以上是根据提供的接口实现信息生成的接口文档，涵盖了类和函数的详细说明及调用示例。