# 接口文档

## 类: `ChatChain`
`ChatChain` 类是一个多阶段自动化协作框架，专为基于大语言模型（LLM）的软件开发流程设计。它通过定义清晰的阶段化工作流，协调不同AI角色（如程序员、测试工程师等）协作完成软件开发任务。

### 初始化方法: `__init__`
```python
def __init__(self,
             config_path: str = None,
             config_phase_path: str = None,
             config_role_path: str = None,
             task_prompt: str = None,
             project_name: str = None,
             org_name: str = None,
             model_type: ModelType = ModelType.GPT_3_5_TURBO,
             code_path: str = None) -> None:
```
#### 参数说明:
- `config_path` (str): ChatChain 配置文件的路径。
- `config_phase_path` (str): 阶段配置文件的路径。
- `config_role_path` (str): 角色配置文件的路径。
- `task_prompt` (str): 用户输入的软件开发任务提示。
- `project_name` (str): 用户输入的软件项目名称。
- `org_name` (str): 用户的组织名称。
- `model_type` (ModelType): 使用的模型类型，默认为 `ModelType.GPT_3_5_TURBO`。
- `code_path` (str): 代码文件的路径。

#### 返回值说明:
无返回值。

#### 范围说明:
初始化 `ChatChain` 实例，加载配置文件，初始化工作流和角色提示。

---

### 方法: `make_recruitment`
```python
def make_recruitment(self):
```
#### 参数说明:
无参数。

#### 返回值说明:
无返回值。

#### 范围说明:
招募所有员工，初始化协作环境中的代理。

---

### 方法: `execute_step`
```python
def execute_step(self, phase_item: dict):
```
#### 参数说明:
- `phase_item` (dict): 单个阶段配置，包含阶段名称和类型等信息。

#### 返回值说明:
无返回值。

#### 范围说明:
执行工作流中的单个阶段，根据阶段类型调用相应的执行方法。

---

### 方法: `execute_chain`
```python
def execute_chain(self):
```
#### 参数说明:
无参数。

#### 返回值说明:
无返回值。

#### 范围说明:
执行整个工作流链，依次执行每个阶段。

---

### 方法: `get_logfilepath`
```python
def get_logfilepath(self):
```
#### 参数说明:
无参数。

#### 返回值说明:
- `start_time` (str): 软件开发开始的时间。
- `log_filepath` (str): 日志文件的路径。

#### 范围说明:
获取日志文件的路径和开始时间。

---

### 方法: `pre_processing`
```python
def pre_processing(self):
```
#### 参数说明:
无参数。

#### 返回值说明:
无返回值。

#### 范围说明:
进行预处理，清理无用文件并记录全局配置设置。

---

### 方法: `post_processing`
```python
def post_processing(self):
```
#### 参数说明:
无参数。

#### 返回值说明:
无返回值。

#### 范围说明:
进行后处理，汇总生产结果并将日志文件移动到软件目录。

---

### 方法: `self_task_improve`
```python
def self_task_improve(self, task_prompt):
```
#### 参数说明:
- `task_prompt` (str): 原始用户查询提示。

#### 返回值说明:
- `revised_task_prompt` (str): 从提示工程师代理返回的改进后的提示。

#### 范围说明:
请求代理改进用户查询提示，以便更好地指导大语言模型进行软件开发。