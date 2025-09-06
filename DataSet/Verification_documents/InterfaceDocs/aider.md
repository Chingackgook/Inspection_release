# 接口文档

## 类：Coder

### 初始化方法：`__init__`

#### 参数说明：
- `main_model`: 主模型实例。
- `io`: 输入输出实例。
- `repo`: 可选，Git 仓库实例。
- `fnames`: 可选，文件名列表。
- `read_only_fnames`: 可选，只读文件名列表。
- `show_diffs`: 可选，是否显示差异。
- `auto_commits`: 可选，是否自动提交。
- `dirty_commits`: 可选，是否允许脏提交。
- `dry_run`: 可选，是否为干运行。
- `map_tokens`: 可选，映射令牌数量。
- `verbose`: 可选，是否详细输出。
- `stream`: 可选，是否流式处理。
- `use_git`: 可选，是否使用 Git。
- `cur_messages`: 可选，当前消息列表。
- `done_messages`: 可选，已完成消息列表。
- `restore_chat_history`: 可选，是否恢复聊天历史。
- `auto_lint`: 可选，是否自动 lint。
- `auto_test`: 可选，是否自动测试。
- `lint_cmds`: 可选，lint 命令字典。
- `test_cmd`: 可选，测试命令。
- `aider_commit_hashes`: 可选，Aider 提交哈希集合。
- `map_mul_no_files`: 可选，映射多文件数量。
- `commands`: 可选，命令实例。
- `summarizer`: 可选，总结器实例。
- `total_cost`: 可选，总成本。
- `analytics`: 可选，分析实例。
- `map_refresh`: 可选，映射刷新策略。
- `cache_prompts`: 可选，是否缓存提示。
- `num_cache_warming_pings`: 可选，缓存预热次数。
- `suggest_shell_commands`: 可选，是否建议 shell 命令。
- `chat_language`: 可选，聊天语言。
- `detect_urls`: 可选，是否检测 URL。
- `ignore_mentions`: 可选，忽略提及集合。
- `file_watcher`: 可选，文件监视器实例。
- `auto_copy_context`: 可选，是否自动复制上下文。
- `auto_accept_architect`: 可选，是否自动接受架构。

#### 返回值说明：
无返回值。

---

### 属性

- `abs_fnames`: 绝对文件名集合。
- `abs_read_only_fnames`: 绝对只读文件名集合。
- `repo`: Git 仓库实例。
- `last_aider_commit_hash`: 上一个 Aider 提交哈希。
- `aider_edited_files`: Aider 编辑的文件集合。
- `last_asked_for_commit_time`: 上次请求提交的时间。
- `repo_map`: 仓库映射实例。
- `functions`: 函数列表。
- `num_exhausted_context_windows`: 耗尽的上下文窗口数量。
- `num_malformed_responses`: 错误响应数量。
- `last_keyboard_interrupt`: 上次键盘中断时间。
- `num_reflections`: 反思次数。
- `max_reflections`: 最大反思次数。
- `edit_format`: 编辑格式。
- `yield_stream`: 是否流式输出。
- `temperature`: 温度设置。
- `auto_lint`: 是否自动 lint。
- `auto_test`: 是否自动测试。
- `test_cmd`: 测试命令。
- `lint_outcome`: lint 结果。
- `test_outcome`: 测试结果。
- `multi_response_content`: 多响应内容。
- `partial_response_content`: 部分响应内容。
- `commit_before_message`: 提交前消息列表。
- `message_cost`: 消息成本。
- `message_tokens_sent`: 发送的消息令牌数量。
- `message_tokens_received`: 接收的消息令牌数量。
- `add_cache_headers`: 是否添加缓存头。
- `cache_warming_thread`: 缓存预热线程。
- `num_cache_warming_pings`: 缓存预热次数。
- `suggest_shell_commands`: 是否建议 shell 命令。
- `detect_urls`: 是否检测 URL。
- `ignore_mentions`: 忽略提及集合。
- `chat_language`: 聊天语言。
- `file_watcher`: 文件监视器实例。

---

### 方法：`create`

#### 参数说明：
- `main_model`: 可选，主模型实例。
- `edit_format`: 可选，编辑格式。
- `io`: 可选，输入输出实例。
- `from_coder`: 可选，源 Coder 实例。
- `summarize_from_coder`: 可选，是否从源 Coder 总结。
- `**kwargs`: 其他关键字参数。

#### 返回值说明：
返回一个新的 Coder 实例。

#### 调用示例：
```python
coder_instance = Coder.create(main_model=my_model, io=my_io)
```

---

### 方法：`clone`

#### 参数说明：
- `**kwargs`: 其他关键字参数。

#### 返回值说明：
返回一个新的 Coder 实例。

#### 调用示例：
```python
new_coder = coder_instance.clone()
```

---

### 方法：`get_announcements`

#### 参数说明：
无参数。

#### 返回值说明：
返回公告信息的列表。

#### 调用示例：
```python
announcements = coder_instance.get_announcements()
```

---

### 方法：`show_announcements`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
```python
coder_instance.show_announcements()
```

---

### 方法：`add_rel_fname`

#### 参数说明：
- `rel_fname`: 相对文件名。

#### 返回值说明：
无返回值。

#### 调用示例：
```python
coder_instance.add_rel_fname("example.py")
```

---

### 方法：`drop_rel_fname`

#### 参数说明：
- `fname`: 文件名。

#### 返回值说明：
返回布尔值，表示是否成功删除。

#### 调用示例：
```python
success = coder_instance.drop_rel_fname("example.py")
```

---

### 方法：`abs_root_path`

#### 参数说明：
- `path`: 路径字符串。

#### 返回值说明：
返回绝对路径。

#### 调用示例：
```python
absolute_path = coder_instance.abs_root_path("example.py")
```

---

### 方法：`show_pretty`

#### 参数说明：
无参数。

#### 返回值说明：
返回布尔值，表示是否显示美观输出。

#### 调用示例：
```python
is_pretty = coder_instance.show_pretty()
```

---

### 方法：`get_abs_fnames_content`

#### 参数说明：
无参数。

#### 返回值说明：
返回绝对文件名及其内容的生成器。

#### 调用示例：
for fname, content in coder_instance.get_abs_fnames_content():
    print(fname, content)

---

### 方法：`choose_fence`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
```python
coder_instance.choose_fence()
```

---

### 方法：`get_files_content`

#### 参数说明：
- `fnames`: 可选，文件名列表。

#### 返回值说明：
返回文件内容的字符串。

#### 调用示例：
files_content = coder_instance.get_files_content()

---

### 方法：`get_read_only_files_content`

#### 参数说明：
无参数。

#### 返回值说明：
返回只读文件内容的字符串。

#### 调用示例：
readonly_content = coder_instance.get_read_only_files_content()

---

### 方法：`get_cur_message_text`

#### 参数说明：
无参数。

#### 返回值说明：
返回当前消息文本的字符串。

#### 调用示例：
cur_text = coder_instance.get_cur_message_text()

---

### 方法：`get_ident_mentions`

#### 参数说明：
- `text`: 输入文本字符串。

#### 返回值说明：
返回识别到的单词集合。

#### 调用示例：
ident_mentions = coder_instance.get_ident_mentions("This is a test.")

---

### 方法：`get_ident_filename_matches`

#### 参数说明：
- `idents`: 单词集合。

#### 返回值说明：
返回匹配的文件名集合。

#### 调用示例：
matches = coder_instance.get_ident_filename_matches({"example", "test"})

---

### 方法：`get_repo_map`

#### 参数说明：
- `force_refresh`: 可选，强制刷新标志。

#### 返回值说明：
返回仓库映射内容。

#### 调用示例：
repo_map = coder_instance.get_repo_map()

---

### 方法：`get_repo_messages`

#### 参数说明：
无参数。

#### 返回值说明：
返回仓库消息列表。

#### 调用示例：
repo_messages = coder_instance.get_repo_messages()

---

### 方法：`get_readonly_files_messages`

#### 参数说明：
无参数。

#### 返回值说明：
返回只读文件消息列表。

#### 调用示例：
readonly_messages = coder_instance.get_readonly_files_messages()

---

### 方法：`get_chat_files_messages`

#### 参数说明：
无参数。

#### 返回值说明：
返回聊天文件消息列表。

#### 调用示例：
chat_files_messages = coder_instance.get_chat_files_messages()

---

### 方法：`get_images_message`

#### 参数说明：
- `fnames`: 文件名列表。

#### 返回值说明：
返回图像消息。

#### 调用示例：
images_message = coder_instance.get_images_message(["image1.png", "image2.jpg"])

---

### 方法：`run_stream`

#### 参数说明：
- `user_message`: 用户消息字符串。

#### 返回值说明：
返回生成器。

#### 调用示例：
for response in coder_instance.run_stream("Hello, world!"):
    print(response)

---

### 方法：`init_before_message`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.init_before_message()

---

### 方法：`run`

#### 参数说明：
- `with_message`: 可选，消息字符串。
- `preproc`: 可选，是否预处理。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.run(with_message="Hello, world!")

---

### 方法：`copy_context`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.copy_context()

---

### 方法：`get_input`

#### 参数说明：
无参数。

#### 返回值说明：
返回用户输入的字符串。

#### 调用示例：
user_input = coder_instance.get_input()

---

### 方法：`preproc_user_input`

#### 参数说明：
- `inp`: 输入字符串。

#### 返回值说明：
返回处理后的字符串。

#### 调用示例：
processed_input = coder_instance.preproc_user_input("Some input text.")

---

### 方法：`run_one`

#### 参数说明：
- `user_message`: 用户消息字符串。
- `preproc`: 可选，是否预处理。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.run_one("Hello, world!", preproc=True)

---

### 方法：`check_and_open_urls`

#### 参数说明：
- `exc`: 异常对象。
- `friendly_msg`: 可选，友好的消息字符串。

#### 返回值说明：
返回 URL 列表。

#### 调用示例：
urls = coder_instance.check_and_open_urls(exception)

---

### 方法：`check_for_urls`

#### 参数说明：
- `inp`: 输入字符串。

#### 返回值说明：
返回处理后的字符串。

#### 调用示例：
processed_input = coder_instance.check_for_urls("Check this link: http://example.com")

---

### 方法：`keyboard_interrupt`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.keyboard_interrupt()

---

### 方法：`summarize_start`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.summarize_start()

---

### 方法：`summarize_worker`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.summarize_worker()

---

### 方法：`summarize_end`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.summarize_end()

---

### 方法：`move_back_cur_messages`

#### 参数说明：
- `message`: 消息字符串。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.move_back_cur_messages("Some message.")

---

### 方法：`get_user_language`

#### 参数说明：
无参数。

#### 返回值说明：
返回用户语言字符串。

#### 调用示例：
user_language = coder_instance.get_user_language()

---

### 方法：`get_platform_info`

#### 参数说明：
无参数。

#### 返回值说明：
返回平台信息字符串。

#### 调用示例：
platform_info = coder_instance.get_platform_info()

---

### 方法：`fmt_system_prompt`

#### 参数说明：
- `prompt`: 提示字符串。

#### 返回值说明：
返回格式化后的提示字符串。

#### 调用示例：
formatted_prompt = coder_instance.fmt_system_prompt("This is a system prompt.")

---

### 方法：`format_chat_chunks`

#### 参数说明：
无参数。

#### 返回值说明：
返回聊天块。

#### 调用示例：
chat_chunks = coder_instance.format_chat_chunks()

---

### 方法：`format_messages`

#### 参数说明：
无参数。

#### 返回值说明：
返回格式化后的消息。

#### 调用示例：
formatted_messages = coder_instance.format_messages()

---

### 方法：`warm_cache`

#### 参数说明：
- `chunks`: 缓存块。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.warm_cache(chunks)

---

### 方法：`check_tokens`

#### 参数说明：
- `messages`: 消息列表。

#### 返回值说明：
返回布尔值，表示是否可以继续。

#### 调用示例：
can_continue = coder_instance.check_tokens(messages)

---

### 方法：`send_message`

#### 参数说明：
- `inp`: 输入字符串。

#### 返回值说明：
返回生成器。

#### 调用示例：
for response in coder_instance.send_message("Hello, world!"):
    print(response)

---

### 方法：`show_send_output`

#### 参数说明：
- `completion`: 完成对象。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.show_send_output(completion)

---

### 方法：`show_send_output_stream`

#### 参数说明：
- `completion`: 完成对象。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.show_send_output_stream(completion)

---

### 方法：`live_incremental_response`

#### 参数说明：
- `final`: 布尔值，表示是否为最终响应。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.live_incremental_response(final=True)

---

### 方法：`render_incremental_response`

#### 参数说明：
- `final`: 布尔值，表示是否为最终响应。

#### 返回值说明：
返回渲染的响应字符串。

#### 调用示例：
response = coder_instance.render_incremental_response(final=True)

---

### 方法：`remove_reasoning_content`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.remove_reasoning_content()

---

### 方法：`calculate_and_show_tokens_and_cost`

#### 参数说明：
- `messages`: 消息列表。
- `completion`: 可选，完成对象。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.calculate_and_show_tokens_and_cost(messages)

---

### 方法：`show_usage_report`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.show_usage_report()

---

### 方法：`get_multi_response_content_in_progress`

#### 参数说明：
- `final`: 可选，布尔值，表示是否为最终响应。

#### 返回值说明：
返回多响应内容字符串。

#### 调用示例：
multi_response = coder_instance.get_multi_response_content_in_progress(final=True)

---

### 方法：`get_rel_fname`

#### 参数说明：
- `fname`: 文件名字符串。

#### 返回值说明：
返回相对文件名字符串。

#### 调用示例：
relative_fname = coder_instance.get_rel_fname("example.py")

---

### 方法：`get_inchat_relative_files`

#### 参数说明：
无参数。

#### 返回值说明：
返回聊天中的相对文件名列表。

#### 调用示例：
inchat_files = coder_instance.get_inchat_relative_files()

---

### 方法：`is_file_safe`

#### 参数说明：
- `fname`: 文件名字符串。

#### 返回值说明：
返回布尔值，表示文件是否安全。

#### 调用示例：
is_safe = coder_instance.is_file_safe("example.py")

---

### 方法：`get_all_relative_files`

#### 参数说明：
无参数。

#### 返回值说明：
返回所有相对文件名列表。

#### 调用示例：
all_files = coder_instance.get_all_relative_files()

---

### 方法：`get_all_abs_files`

#### 参数说明：
无参数。

#### 返回值说明：
返回所有绝对文件名列表。

#### 调用示例：
all_abs_files = coder_instance.get_all_abs_files()

---

### 方法：`get_addable_relative_files`

#### 参数说明：
无参数。

#### 返回值说明：
返回可添加的相对文件名列表。

#### 调用示例：
addable_files = coder_instance.get_addable_relative_files()

---

### 方法：`check_for_dirty_commit`

#### 参数说明：
- `path`: 文件路径字符串。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.check_for_dirty_commit("example.py")

---

### 方法：`allowed_to_edit`

#### 参数说明：
- `path`: 文件路径字符串。

#### 返回值说明：
返回布尔值，表示是否允许编辑。

#### 调用示例：
can_edit = coder_instance.allowed_to_edit("example.py")

---

### 方法：`check_added_files`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.check_added_files()

---

### 方法：`prepare_to_edit`

#### 参数说明：
- `edits`: 编辑列表。

#### 返回值说明：
返回准备好的编辑列表。

#### 调用示例：
prepared_edits = coder_instance.prepare_to_edit(edits)

---

### 方法：`apply_updates`

#### 参数说明：
无参数。

#### 返回值说明：
返回编辑的集合。

#### 调用示例：
edited_files = coder_instance.apply_updates()

---

### 方法：`parse_partial_args`

#### 参数说明：
无参数。

#### 返回值说明：
返回解析后的参数字典。

#### 调用示例：
args = coder_instance.parse_partial_args()

---

### 方法：`get_context_from_history`

#### 参数说明：
- `history`: 历史消息列表。

#### 返回值说明：
返回上下文字符串。

#### 调用示例：
context = coder_instance.get_context_from_history(history)

---

### 方法：`auto_commit`

#### 参数说明：
- `edited`: 编辑的文件列表。
- `context`: 可选，上下文字符串。

#### 返回值说明：
返回提交消息字符串。

#### 调用示例：
commit_message = coder_instance.auto_commit(edited_files)

---

### 方法：`show_auto_commit_outcome`

#### 参数说明：
- `res`: 提交结果。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.show_auto_commit_outcome(commit_result)

---

### 方法：`show_undo_hint`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.show_undo_hint()

---

### 方法：`dirty_commit`

#### 参数说明：
无参数。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.dirty_commit()

---

### 方法：`get_edits`

#### 参数说明：
- `mode`: 可选，模式字符串。

#### 返回值说明：
返回编辑列表。

#### 调用示例：
edits = coder_instance.get_edits()

---

### 方法：`apply_edits`

#### 参数说明：
- `edits`: 编辑列表。

#### 返回值说明：
无返回值。

#### 调用示例：
coder_instance.apply_edits(edits)

---

### 方法：`apply_edits_dry_run`

#### 参数说明：
- `edits`: 编辑列表。

#### 返回值说明：
返回编辑列表。

#### 调用示例：
dry_run_edits = coder_instance.apply_edits_dry_run(edits)

---

### 方法：`run_shell_commands`

#### 参数说明：
无参数。

#### 返回值说明：
返回命令输出字符串。

#### 调用示例：
output = coder_instance.run_shell_commands()

---

### 方法：`handle_shell_commands`

#### 参数说明：
- `commands_str`: 命令字符串。
- `group`: 确认组实例。

#### 返回值说明：
返回命令输出字符串。

#### 调用示例：
output = coder_instance.handle_shell_commands("ls -la", group)

---

以上是 `Coder` 类的接口文档，涵盖了初始化信息、属性、方法及其参数、返回值说明和调用示例。