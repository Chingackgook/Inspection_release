为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式，并提取出所需的参数。以下是对源代码的分析和替换方案：

### 1. 分析源代码中的函数调用

#### a. `afrom_files`
- **调用**: `brain = await Brain.afrom_files(name="test_brain", file_paths=[temp_file.name])`
- **参数**:
  - `name`: "test_brain"
  - `file_paths`: `[temp_file.name]` (临时文件的路径)

#### b. `save`
- **调用**: `await brain.save("~/.local/quivr")`
- **参数**:
  - `folder_path`: `"~/.local/quivr"`

#### c. `ask_streaming`
- **调用**: 
  ```python
  async for chunk in brain.ask_streaming(question, rag_pipeline=QuivrQARAG):
  ```
- **参数**:
  - `question`: `"what is gold? answer in french"`
  - `rag_pipeline`: `QuivrQARAG`

- **调用**: 
  ```python
  async for chunk in brain.ask_streaming(question, rag_pipeline=QuivrQARAGLangGraph):
  ```
- **参数**:
  - `question`: `"what is gold? answer in french"`
  - `rag_pipeline`: `QuivrQARAGLangGraph`

### 2. 替换为 `exe.run("function_name", **kwargs)` 的形式

根据上述分析，我们可以将源代码中的函数调用替换为 `exe.run` 的形式。以下是替换后的方案：

```python
async def main():
    dotenv_path = "/Users/jchevall/Coding/QuivrHQ/quivr/.env"
    load_dotenv(dotenv_path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as temp_file:
        temp_file.write("Gold is a liquid of blue-like colour.")
        temp_file.flush()

        # 替换 afrom_files 调用
        brain = await exe.run("afrom_files", name="test_brain", file_paths=[temp_file.name])

        # 替换 save 调用
        await exe.run("save", folder_path="~/.local/quivr")

        question = "what is gold? answer in french"
        
        # 替换 ask_streaming 调用 (QuivrQARAG)
        async for chunk in exe.run("ask_streaming", question=question, rag_pipeline=QuivrQARAG):
            print("answer QuivrQARAG:", chunk.answer)

        # 替换 ask_streaming 调用 (QuivrQARAGLangGraph)
        async for chunk in exe.run("ask_streaming", question=question, rag_pipeline=QuivrQARAGLangGraph):
            print("answer QuivrQARAGLangGraph:", chunk.answer)
```

### 3. 模拟输入参数

在替换过程中，我们需要确保所有参数都能正确传递。以下是对每个函数参数的模拟输入分析：

- **afrom_files**:
  - `name`: 直接使用字符串 `"test_brain"`。
  - `file_paths`: 使用临时文件的路径 `temp_file.name`。

- **save**:
  - `folder_path`: 使用字符串 `"~/.local/quivr"`。

- **ask_streaming**:
  - `question`: 使用字符串 `"what is gold? answer in french"`。
  - `rag_pipeline`: 使用 `QuivrQARAG` 和 `QuivrQARAGLangGraph` 作为参数。

### 4. 方案总结

通过上述分析和替换，我们将源代码中的关键函数调用成功替换为 `exe.run("function_name", **kwargs)` 的形式，并确保所有参数都能正确传递。这样做的好处是可以将函数调用统一化，便于后续的维护和扩展。