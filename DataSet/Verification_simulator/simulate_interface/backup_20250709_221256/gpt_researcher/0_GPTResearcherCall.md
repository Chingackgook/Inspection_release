为了将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，我们需要逐一分析每个函数的调用方式，并确定所需的参数。以下是对每个关键函数的分析和替换方案：

### 1. `conduct_research`
- **调用方式**: `context = await researcher.conduct_research(on_progress=on_progress)`
- **替换为**: `context = await exe.run("conduct_research", on_progress=on_progress)`
- **参数分析**: `on_progress` 是一个回调函数，用于跟踪研究进度。

### 2. `write_report`
- **调用方式**: `report = await researcher.write_report()`
- **替换为**: `report = await exe.run("write_report")`
- **参数分析**: 无需额外参数。

### 3. `write_report_conclusion`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告后使用。
- **替换为**: `conclusion = await exe.run("write_report_conclusion", report_body=report)`
- **参数分析**: `report_body` 是生成的报告内容。

### 4. `write_introduction`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告前使用。
- **替换为**: `intro = await exe.run("write_introduction")`
- **参数分析**: 无需额外参数。

### 5. `quick_search`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在研究过程中使用。
- **替换为**: `results = await exe.run("quick_search", query=query, query_domains=query_domains)`
- **参数分析**: `query` 是查询字符串，`query_domains` 是可选的查询域名列表。

### 6. `get_subtopics`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在研究过程中使用。
- **替换为**: `subtopics = await exe.run("get_subtopics")`
- **参数分析**: 无需额外参数。

### 7. `get_draft_section_titles`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `draft_section_titles = await exe.run("get_draft_section_titles", current_subtopic=current_subtopic)`
- **参数分析**: `current_subtopic` 是当前的子主题。

### 8. `get_similar_written_contents_by_draft_section_titles`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `similar_contents = await exe.run("get_similar_written_contents_by_draft_section_titles", current_subtopic=current_subtopic, draft_section_titles=draft_section_titles, written_contents=written_contents, max_results=max_results)`
- **参数分析**: 需要提供 `current_subtopic`、`draft_section_titles`、`written_contents` 和 `max_results`。

### 9. `get_research_images`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `images = await exe.run("get_research_images", top_k=top_k)`
- **参数分析**: `top_k` 是返回的图像数量。

### 10. `add_research_images`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `await exe.run("add_research_images", images=images)`
- **参数分析**: `images` 是要添加的图像列表。

### 11. `get_research_sources`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `sources = await exe.run("get_research_sources")`
- **参数分析**: 无需额外参数。

### 12. `add_research_sources`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `await exe.run("add_research_sources", sources=sources)`
- **参数分析**: `sources` 是要添加的来源列表。

### 13. `add_references`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `updated_markdown = await exe.run("add_references", report_markdown=report, visited_urls=visited_urls)`
- **参数分析**: `report_markdown` 是报告的 Markdown 文本，`visited_urls` 是已访问的 URL 集合。

### 14. `extract_headers`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `headers = await exe.run("extract_headers", markdown_text=report)`
- **参数分析**: `markdown_text` 是报告的 Markdown 文本。

### 15. `extract_sections`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `sections = await exe.run("extract_sections", markdown_text=report)`
- **参数分析**: `markdown_text` 是报告的 Markdown 文本。

### 16. `table_of_contents`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `toc = await exe.run("table_of_contents", markdown_text=report)`
- **参数分析**: `markdown_text` 是报告的 Markdown 文本。

### 17. `get_source_urls`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `urls = await exe.run("get_source_urls")`
- **参数分析**: 无需额外参数。

### 18. `get_research_context`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `context = await exe.run("get_research_context")`
- **参数分析**: 无需额外参数。

### 19. `get_costs`
- **调用方式**: `costs = researcher.get_costs()`
- **替换为**: `costs = await exe.run("get_costs")`
- **参数分析**: 无需额外参数。

### 20. `set_verbose`
- **调用方式**: `researcher.set_verbose(True)`
- **替换为**: `await exe.run("set_verbose", verbose=True)`
- **参数分析**: `verbose` 是布尔值，指示是否详细输出。

### 21. `add_costs`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在生成报告时使用。
- **替换为**: `await exe.run("add_costs", cost=cost)`
- **参数分析**: `cost` 是要添加的成本。

### 22. `_log_event`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在研究过程中使用。
- **替换为**: `await exe.run("_log_event", event_type=event_type, **kwargs)`
- **参数分析**: `event_type` 是事件类型，`**kwargs` 是其他关键字参数。

### 23. `_handle_deep_research`
- **调用方式**: 该函数在示例代码中未直接调用，但可以在研究过程中使用。
- **替换为**: `context = await exe.run("_handle_deep_research", on_progress=on_progress)`
- **参数分析**: `on_progress` 是进度回调函数。

### 模拟输入方案
在替换函数时，我们需要确保所有参数都能正确获取。以下是模拟输入的方案：

1. **query**: 从主函数的输入获取，例如 `"What are the most effective ways for beginners to start investing?"`。
2. **on_progress**: 定义一个进度回调函数，打印进度信息。
3. **report**: 在调用 `write_report` 后生成的报告内容。
4. **visited_urls**: 可以初始化为空集合或从其他来源获取。
5. **current_subtopic**: 可以从 `get_subtopics` 的返回值中获取。
6. **max_results**: 可以设置为一个合理的默认值，例如 10。
7. **top_k**: 可以设置为一个合理的默认值，例如 10。
8. **cost**: 可以根据研究的复杂性和时间来估算。

通过以上分析和方案，我们可以将源代码中的关键函数替换为 `exe.run("function_name", **kwargs)` 的形式，并确保所有参数都能正确传递。