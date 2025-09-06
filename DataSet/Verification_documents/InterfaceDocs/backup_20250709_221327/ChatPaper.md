# 接口文档

## 类：Reader

### 初始化方法：`__init__`
- **函数名**: `__init__`
- **参数说明**:
  - `key_word` (str): 读者感兴趣的关键词。
  - `query` (str): 读者输入的搜索查询。
  - `filter_keys` (str): 用于在摘要中筛选的关键词。
  - `root_path` (str, 默认值='./'): 根路径。
  - `gitee_key` (str, 默认值=''): Gitee API 密钥。
  - `sort` (arxiv.SortCriterion): 读者选择的排序方式。
  - `user_name` (str, 默认值='defualt'): 读者姓名。
  - `args` (argparse.Namespace): 命令行参数。
- **返回值**: 无
- **范围说明**: 初始化Reader类的实例，设置相关属性和配置。


### 方法：`summary_with_chat`
- **函数名**: `summary_with_chat`
- **参数说明**:
  - `paper_list` (list): 论文对象列表。
- **返回值**: 无
- **范围说明**: 使用聊天模型对论文进行总结。

### 方法：`chat_conclusion`
- **函数名**: `chat_conclusion`
- **参数说明**:
  - `text` (str): 输入文本。
  - `conclusion_prompt_token` (int, 默认值=800): 结论提示的token数量。
- **返回值**: `str` - 生成的结论文本。
- **范围说明**: 与聊天模型交互以生成论文结论。

### 方法：`chat_method`
- **函数名**: `chat_method`
- **参数说明**:
  - `text` (str): 输入文本。
  - `method_prompt_token` (int, 默认值=800): 方法提示的token数量。
- **返回值**: `str` - 生成的方法文本。
- **范围说明**: 与聊天模型交互以生成论文方法部分。

### 方法：`chat_summary`
- **函数名**: `chat_summary`
- **参数说明**:
  - `text` (str): 输入文本。
  - `summary_prompt_token` (int, 默认值=1100): 摘要提示的token数量。
- **返回值**: `str` - 生成的摘要文本。
- **范围说明**: 与聊天模型交互以生成论文摘要。


## 示例调用
```python
# 创建Reader实例
reader = Reader(key_word='Machine Learning', query='deep learning', filter_keys='neural network', args=argparse.Namespace(language='en', file_format='md', save_image=False))

# 获取arXiv搜索结果
search_results = reader.get_arxiv(max_results=10)

# 筛选arXiv结果
filtered_results = reader.filter_arxiv(max_results=10)

# 下载PDF
papers = reader.download_pdf(filtered_results)

# 使用聊天模型进行总结
reader.summary_with_chat(papers)
```