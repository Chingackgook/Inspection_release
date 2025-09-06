# 接口文档

## 1. 类：Reader
### 描述
Reader类是一个用于处理学术论文的接口类，主要功能包括与大模型进行交互以生成论文的总结、方法和结论。该类还提供了从arXiv获取论文、过滤论文、下载PDF、上传图片等功能。

### 属性
- `user_name`: 读者姓名
- `key_word`: 读者感兴趣的关键词
- `query`: 读者输入的搜索查询
- `sort`: 读者选择的排序方式
- `language`: 读者选择的语言（英语或中文）
- `filter_keys`: 用于在摘要中筛选的关键词
- `root_path`: 文件存储的根路径
- `config`: 配置文件解析器
- `chat_api_list`: OpenAI API密钥列表
- `chatgpt_model`: 使用的ChatGPT模型
- `cur_api`: 当前使用的API索引
- `file_format`: 文件保存格式
- `gitee_key`: Gitee API密钥
- `max_token_num`: 最大token数量
- `encoding`: 编码方式

### 方法
#### 1. `__init__(self, key_word, query, filter_keys, root_path='./', gitee_key='', sort=arxiv.SortCriterion.SubmittedDate, user_name='defualt', args=None)`
- **参数说明**:
  - `key_word`: 读者感兴趣的关键词
  - `query`: 读者输入的搜索查询
  - `filter_keys`: 用于在摘要中筛选的关键词
  - `root_path`: 文件存储的根路径，默认为'./'
  - `gitee_key`: Gitee API密钥，默认为空
  - `sort`: 排序方式，默认为提交日期
  - `user_name`: 读者姓名，默认为'defualt'
  - `args`: 其他参数
- **返回值**: 无
- **作用**: 初始化Reader类的实例，设置相关属性并读取配置文件。

#### 2. `get_arxiv(self, max_results=30)`
- **参数说明**:
  - `max_results`: 最大返回结果数量，默认为30
- **返回值**: 返回arxiv搜索对象
- **作用**: 根据查询获取arXiv的搜索结果。

#### 3. `filter_arxiv(self, max_results=30)`
- **参数说明**:
  - `max_results`: 最大返回结果数量，默认为30
- **返回值**: 返回过滤后的论文结果列表
- **作用**: 过滤arXiv搜索结果，确保每个关键词都能在摘要中找到。

#### 4. `validateTitle(self, title)`
- **参数说明**:
  - `title`: 论文标题
- **返回值**: 返回修正后的标题
- **作用**: 修正论文标题中的非法字符。

#### 5. `download_pdf(self, filter_results)`
- **参数说明**:
  - `filter_results`: 过滤后的论文结果列表
- **返回值**: 返回下载的论文对象列表
- **作用**: 下载过滤后的论文PDF文件并解析。

#### 6. `try_download_pdf(self, result, path, pdf_name)`
- **参数说明**:
  - `result`: 论文结果对象
  - `path`: 下载路径
  - `pdf_name`: PDF文件名
- **返回值**: 无
- **作用**: 尝试下载PDF文件，带有重试机制。

#### 7. `upload_gitee(self, image_path, image_name='', ext='png')`
- **参数说明**:
  - `image_path`: 图片文件路径
  - `image_name`: 图片文件名，默认为空
  - `ext`: 图片文件扩展名，默认为'png'
- **返回值**: 返回上传后图片的URL
- **作用**: 将图片上传到Gitee并返回其下载链接。

#### 8. `summary_with_chat(self, paper_list)`
- **参数说明**:
  - `paper_list`: 论文对象列表
- **返回值**: 无
- **作用**: 与大模型交互，生成论文的总结、方法和结论，并保存到文件。

#### 9. `chat_conclusion(self, text, conclusion_prompt_token=800)`
- **参数说明**:
  - `text`: 输入文本
  - `conclusion_prompt_token`: 用于生成结论的token数量，默认为800
- **返回值**: 返回生成的结论文本
- **作用**: 与大模型交互，生成论文的结论部分。

#### 10. `chat_method(self, text, method_prompt_token=800)`
- **参数说明**:
  - `text`: 输入文本
  - `method_prompt_token`: 用于生成方法的token数量，默认为800
- **返回值**: 返回生成的方法文本
- **作用**: 与大模型交互，生成论文的方法部分。

#### 11. `chat_summary(self, text, summary_prompt_token=1100)`
- **参数说明**:
  - `text`: 输入文本
  - `summary_prompt_token`: 用于生成摘要的token数量，默认为1100
- **返回值**: 返回生成的摘要文本
- **作用**: 与大模型交互，生成论文的摘要部分。

#### 12. `export_to_markdown(self, text, file_name, mode='w')`
- **参数说明**:
  - `text`: 要写入文件的文本
  - `file_name`: 文件名
  - `mode`: 文件打开模式，默认为'w'
- **返回值**: 无
- **作用**: 将文本导出为Markdown格式并保存到指定文件。

#### 13. `show_info(self)`
- **参数说明**: 无
- **返回值**: 无
- **作用**: 打印读者的关键信息，包括关键词、查询和排序方式。