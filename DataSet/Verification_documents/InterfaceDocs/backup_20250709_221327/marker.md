# 接口文档

## 类：ConfigParser

### 初始化方法

#### `__init__(self, cli_options: dict)`

- **参数说明**：
  - `cli_options`: 字典类型，包含从命令行输入获取的选项和参数。

- **返回值说明**：
  - 无返回值。

### 方法

#### `common_options(fn)`

- **参数说明**：
  - `fn`: 被装饰的方法，通常为命令行接口的点击命令的函数。

- **返回值说明**：
  - 返回经过增强的函数，添加了多个配置选项。

- **范围说明**：
  - 无特殊要求。

- **示例**：
  ```python
  @ConfigParser.common_options
  def cli_command():
      pass
  ```

#### `generate_config_dict(self) -> Dict[str, any]`

- **参数说明**：
  - 无参数。

- **返回值说明**：
  - 返回一个字典，包含程序运行所需的配置。

- **范围说明**：
  - 返回的字典配置包括debug模式、页面范围、语言、是否禁用多线程等配置项。

- **示例**：
  ```python
  parser = ConfigParser(cli_options)
  config = parser.generate_config_dict()
  ```

#### `get_llm_service(self)`

- **参数说明**：
  - 无参数。

- **返回值说明**：
  - 返回已配置的LLM服务类的全名，如果未启用LLM，则返回`None`。

- **范围说明**：
  - 返回值取决于`use_llm`选项的配置。

- **示例**：
  ```python
  llm_service = parser.get_llm_service()
  ```

#### `get_renderer(self)`

- **参数说明**：
  - 无参数。

- **返回值说明**：
  - 返回与所选输出格式对应的渲染类的名称。

- **范围说明**：
  - 输出格式必须是`json`、`markdown`或`html`。

- **示例**：
  ```python
  renderer = parser.get_renderer()
  ```

#### `get_processors(self)`

- **参数说明**：
  - 无参数。

- **返回值说明**：
  - 返回处理器类的列表，如果未配置，则返回`None`。

- **范围说明**：
  - 如指定的处理器无法加载，将抛出异常。

- **示例**：
  ```python
  processors = parser.get_processors()
  ```

#### `get_converter_cls(self)`

- **参数说明**：
  - 无参数。

- **返回值说明**：
  - 返回转换器类，如果未指定则返回默认的`PdfConverter`。

- **范围说明**：
  - 如果指定的转换器无法加载，将抛出异常。

- **示例**：
  ```python
  converter_cls = parser.get_converter_cls()
  ```

#### `get_output_folder(self, filepath: str)`

- **参数说明**：
  - `filepath`: 字符串类型，文件的路径。

- **返回值说明**：
  - 返回输出文件夹的路径。

- **范围说明**：
  - 根据文件名生成对应的输出目录，如果目录不存在则创建。

- **示例**：
  ```python
  output_folder = parser.get_output_folder("example.pdf")
  ```

#### `get_base_filename(self, filepath: str)`

- **参数说明**：
  - `filepath`: 字符串类型，文件的路径。

- **返回值说明**：
  - 返回基础文件名，不包括扩展名。

- **范围说明**：
  - 返回值为文件的基本名称。

- **示例**：
  ```python
  base_filename = parser.get_base_filename("example.pdf")
  ```

## 额外说明

### 引用的类和函数
- `JSONRenderer`, `MarkdownRenderer`, `HTMLRenderer`, `PdfConverter`, `settings` 及其他未定义的类和方法为外部依赖，应根据具体实现提供详细文档。