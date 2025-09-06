# 接口文档

## 类：`GPTResearcher`

### 初始化方法：`__init__`

#### 参数说明：
- `query` (str): 研究查询的字符串。
- `report_type` (str, optional): 报告类型，默认为 `ReportType.ResearchReport.value`。
- `report_format` (str, optional): 报告格式，默认为 "markdown"。
- `report_source` (str, optional): 报告来源，默认为 `ReportSource.Web.value`。
- `tone` (Tone, optional): 报告语气，默认为 `Tone.Objective`。
- `source_urls` (list[str], optional): 来源 URL 列表。
- `document_urls` (list[str], optional): 文档 URL 列表。
- `complement_source_urls` (bool, optional): 是否补充来源 URL，默认为 False。
- `query_domains` (list[str], optional): 查询域名列表。
- `documents` (optional): 相关文档。
- `vector_store` (optional): 向量存储。
- `vector_store_filter` (optional): 向量存储过滤器。
- `config_path` (optional): 配置文件路径。
- `websocket` (optional): WebSocket 连接。
- `agent` (optional): 代理。
- `role` (optional): 角色。
- `parent_query` (str, optional): 父查询，默认为空字符串。
- `subtopics` (list, optional): 子主题列表。
- `visited_urls` (set, optional): 已访问 URL 集合。
- `verbose` (bool, optional): 是否详细输出，默认为 True。
- `context` (optional): 上下文。
- `headers` (dict, optional): 请求头。
- `max_subtopics` (int, optional): 最大子主题数，默认为 5。
- `log_handler` (optional): 日志处理器。
- `prompt_family` (str, optional): 提示家族，默认为配置中的提示家族。

#### 返回值说明：
无返回值。

---

### 方法：`_log_event`

#### 参数说明：
- `event_type` (str): 事件类型。
- `**kwargs`: 其他关键字参数，具体取决于事件类型。

#### 返回值说明：
无返回值。

---

### 方法：`conduct_research`

#### 参数说明：
- `on_progress` (optional): 进度回调函数。

#### 返回值说明：
- `context` (list): 研究上下文。

---

### 方法：`_handle_deep_research`

#### 参数说明：
- `on_progress` (optional): 进度回调函数。

#### 返回值说明：
- `context` (list): 深度研究上下文。

---

### 方法：`write_report`

#### 参数说明：
- `existing_headers` (list, optional): 已存在的标题列表。
- `relevant_written_contents` (list, optional): 相关已写内容列表。
- `ext_context` (optional): 外部上下文。
- `custom_prompt` (str, optional): 自定义提示。

#### 返回值说明：
- `report` (str): 生成的报告。

---

### 方法：`write_report_conclusion`

#### 参数说明：
- `report_body` (str): 报告主体。

#### 返回值说明：
- `conclusion` (str): 报告结论。

---

### 方法：`write_introduction`

#### 参数说明：
无参数。

#### 返回值说明：
- `intro` (str): 报告引言。

---

### 方法：`quick_search`

#### 参数说明：
- `query` (str): 查询字符串。
- `query_domains` (list[str], optional): 查询域名列表。

#### 返回值说明：
- `results` (list): 搜索结果列表。

---

### 方法：`get_subtopics`

#### 参数说明：
无参数。

#### 返回值说明：
- `subtopics` (list): 子主题列表。

---

### 方法：`get_draft_section_titles`

#### 参数说明：
- `current_subtopic` (str): 当前子主题。

#### 返回值说明：
- `draft_section_titles` (list): 草稿章节标题列表。

---

### 方法：`get_similar_written_contents_by_draft_section_titles`

#### 参数说明：
- `current_subtopic` (str): 当前子主题。
- `draft_section_titles` (list[str]): 草稿章节标题列表。
- `written_contents` (list[dict]): 已写内容列表。
- `max_results` (int, optional): 最大结果数，默认为 10。

#### 返回值说明：
- `similar_contents` (list[str]): 相似已写内容列表。

---

### 方法：`get_research_images`

#### 参数说明：
- `top_k` (int, optional): 返回的图像数量，默认为 10。

#### 返回值说明：
- `images` (list[dict[str, Any]]): 研究图像列表。

---

### 方法：`add_research_images`

#### 参数说明：
- `images` (list[dict[str, Any]]): 要添加的图像列表。

#### 返回值说明：
无返回值。

---

### 方法：`get_research_sources`

#### 参数说明：
无参数。

#### 返回值说明：
- `sources` (list[dict[str, Any]]): 研究来源列表。

---

### 方法：`add_research_sources`

#### 参数说明：
- `sources` (list[dict[str, Any]]): 要添加的来源列表。

#### 返回值说明：
无返回值。

---

### 方法：`add_references`

#### 参数说明：
- `report_markdown` (str): 报告的 Markdown 文本。
- `visited_urls` (set): 已访问的 URL 集合。

#### 返回值说明：
- `updated_markdown` (str): 更新后的 Markdown 文本。

---

### 方法：`extract_headers`

#### 参数说明：
- `markdown_text` (str): Markdown 文本。

#### 返回值说明：
- `headers` (list[dict]): 提取的标题列表。

---

### 方法：`extract_sections`

#### 参数说明：
- `markdown_text` (str): Markdown 文本。

#### 返回值说明：
- `sections` (list[dict]): 提取的章节列表。

---

### 方法：`table_of_contents`

#### 参数说明：
- `markdown_text` (str): Markdown 文本。

#### 返回值说明：
- `toc` (str): 生成的目录。

---

### 方法：`get_source_urls`

#### 参数说明：
无参数。

#### 返回值说明：
- `urls` (list): 来源 URL 列表。

---

### 方法：`get_research_context`

#### 参数说明：
无参数。

#### 返回值说明：
- `context` (list): 研究上下文。

---

### 方法：`get_costs`

#### 参数说明：
无参数。

#### 返回值说明：
- `costs` (float): 研究成本。

---

### 方法：`set_verbose`

#### 参数说明：
- `verbose` (bool): 是否详细输出。

#### 返回值说明：
无返回值。

---

### 方法：`add_costs`

#### 参数说明：
- `cost` (float): 要添加的成本。

#### 返回值说明：
无返回值。

---

## 调用示例

```python
# 创建 GPTResearcher 实例
researcher = GPTResearcher(
    query="人工智能的未来",
    report_type="ResearchReport",
    report_format="markdown",
    tone=Tone.Objective,
    verbose=True
)

# 开展研究
context = await researcher.conduct_research()

# 写报告
report = await researcher.write_report()

# 获取研究成本
costs = researcher.get_costs()
```