# 接口文档

## 类：`Brain`

### 初始化方法：`__init__`
#### 参数说明：
- `name` (str): Brain的名称。
- `llm` (LLMEndpoint): 用于生成答案的语言模型。
- `id` (UUID | None): Brain的唯一标识符，默认为None。
- `vector_db` (VectorStore | None): 用于存储处理文件的向量存储，默认为None。
- `embedder` (Embeddings | None): 用于创建处理文件索引的嵌入器，默认为None。
- `storage` (StorageBase | None): 用于存储文件的存储方式，默认为None。
- `user_id` (UUID | None): 用户的唯一标识符，默认为None。
- `chat_id` (UUID | None): 聊天的唯一标识符，默认为None。

#### 返回值说明：
无返回值。

---

### 方法：`__repr__`
#### 返回值说明：
- (str): Brain对象的字符串表示。

---

### 方法：`print_info`
#### 返回值说明：
无返回值。

#### 示例：
```python
brain.print_info()
```

---

### 类方法：`load`
#### 参数说明：
- `folder_path` (str | Path): 包含Brain的文件夹路径。

#### 返回值说明：
- (Brain): 从文件夹路径加载的Brain对象。

#### 示例：
```python
brain_loaded = Brain.load("path/to/brain")
brain_loaded.print_info()
```

---

### 异步方法：`save`
#### 参数说明：
- `folder_path` (str | Path): 将Brain保存到的文件夹路径。

#### 返回值说明：
- (str): 保存Brain的文件夹路径。

#### 示例：
```python
await brain.save("path/to/brain")
```

---

### 方法：`info`
#### 返回值说明：
- (BrainInfo): Brain的相关信息。

---

### 属性：`chat_history`
#### 返回值说明：
- (ChatHistory): 默认聊天历史。

---

### 类方法：`afrom_files`
#### 参数说明：
- `name` (str): Brain的名称。
- `file_paths` (list[str | Path]): 要添加到Brain的文件路径列表。
- `vector_db` (VectorStore | None): 用于存储处理文件的向量存储，默认为None。
- `storage` (StorageBase): 用于存储文件的存储方式，默认为TransparentStorage。
- `llm` (LLMEndpoint | None): 用于生成答案的语言模型，默认为None。
- `embedder` (Embeddings | None): 用于创建处理文件索引的嵌入器，默认为None。
- `skip_file_error` (bool): 是否跳过无法处理的文件，默认为False。
- `processor_kwargs` (dict[str, Any] | None): 处理器的附加参数，默认为None。

#### 返回值说明：
- (Brain): 从文件路径创建的Brain对象。

#### 示例：
```python
brain = await Brain.afrom_files(name="My Brain", file_paths=["file1.pdf", "file2.pdf"])
brain.print_info()
```

---

### 类方法：`from_files`
#### 参数说明：
- `name` (str): Brain的名称。
- `file_paths` (list[str | Path]): 要添加到Brain的文件路径列表。
- `vector_db` (VectorStore | None): 用于存储处理文件的向量存储，默认为None。
- `storage` (StorageBase): 用于存储文件的存储方式，默认为TransparentStorage。
- `llm` (LLMEndpoint | None): 用于生成答案的语言模型，默认为None。
- `embedder` (Embeddings | None): 用于创建处理文件索引的嵌入器，默认为None。
- `skip_file_error` (bool): 是否跳过无法处理的文件，默认为False。
- `processor_kwargs` (dict[str, Any] | None): 处理器的附加参数，默认为None。

#### 返回值说明：
- (Self): 从文件路径创建的Brain对象。

---

### 异步方法：`asearch`
#### 参数说明：
- `query` (str | Document): 要搜索的查询。
- `n_results` (int): 返回的结果数量，默认为5。
- `filter` (Callable | Dict[str, Any] | None): 应用到搜索的过滤器，默认为None。
- `fetch_n_neighbors` (int): 要获取的邻居数量，默认为20。

#### 返回值说明：
- (list[SearchResult]): 检索到的文档列表。

#### 示例：
```python
results = await brain.asearch("Why everybody loves Quivr?")
for result in results:
    print(result.chunk.page_content)
```

---

### 方法：`get_chat_history`
#### 参数说明：
- `chat_id` (UUID): 聊天的唯一标识符。

#### 返回值说明：
- (ChatHistory): 指定聊天ID的聊天历史。

---

### 方法：`add_file`
#### 返回值说明：
无返回值。

---

### 异步方法：`ask_streaming`
#### 参数说明：
- `question` (str): 要询问的问题。
- `run_id` (UUID): 运行的唯一标识符。
- `system_prompt` (str | None): 系统提示，默认为None。
- `retrieval_config` (RetrievalConfig | None): 检索配置，默认为None。
- `rag_pipeline` (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): 使用的RAG管道，默认为None。
- `list_files` (list[QuivrKnowledge] | None): 要包含在RAG管道中的文件列表，默认为None。
- `chat_history` (ChatHistory | None): 要使用的聊天历史，默认为None。
- `**input_kwargs`: 其他输入参数。

#### 返回值说明：
- (AsyncGenerator[ParsedRAGChunkResponse, ParsedRAGChunkResponse]): 流式生成的答案。

#### 示例：
```python
async for chunk in brain.ask_streaming("What is the meaning of life?"):
    print(chunk.answer)
```

---

### 异步方法：`aask`
#### 参数说明：
- `run_id` (UUID): 运行的唯一标识符。
- `question` (str): 要询问的问题。
- `system_prompt` (str | None): 系统提示，默认为None。
- `retrieval_config` (RetrievalConfig | None): 检索配置，默认为None。
- `rag_pipeline` (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): 使用的RAG管道，默认为None。
- `list_files` (list[QuivrKnowledge] | None): 要包含在RAG管道中的文件列表，默认为None。
- `chat_history` (ChatHistory | None): 要使用的聊天历史，默认为None。
- `**input_kwargs`: 其他输入参数。

#### 返回值说明：
- (ParsedRAGResponse): 生成的答案。

---

### 方法：`ask`
#### 参数说明：
- `run_id` (UUID): 运行的唯一标识符。
- `question` (str): 要询问的问题。
- `system_prompt` (str | None): 系统提示，默认为None。
- `retrieval_config` (RetrievalConfig | None): 检索配置，默认为None。
- `rag_pipeline` (Type[Union[QuivrQARAG, QuivrQARAGLangGraph]] | None): 使用的RAG管道，默认为None。
- `list_files` (list[QuivrKnowledge] | None): 要包含在RAG管道中的文件列表，默认为None。
- `chat_history` (ChatHistory | None): 要使用的聊天历史，默认为None。

#### 返回值说明：
- (ParsedRAGResponse): 生成的答案。

---

以上是`Brain`类及其方法的接口文档，涵盖了初始化信息、属性、方法及其参数和返回值说明。