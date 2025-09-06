# 接口文档

## 类：Classifier

### 初始化信息
- **构造函数**: `__init__(self, ...)`
  - **参数**: 
    - `...` (具体参数未提供，需根据实际实现补充)
  - **返回值**: 无
  - **范围**: 用于初始化Classifier类的实例。

### 属性
- **属性**: `model`
  - **类型**: `Any`
  - **描述**: 存储加载的模型。

### 方法

#### 方法：evaluate
- **函数名**: `evaluate`
- **参数**:
  - `data_points: Union[list[DT], Dataset]`: 输入的数据点，可以是数据点列表或Dataset对象。
  - `gold_label_type: str`: 真实标签的类型。
  - `out_path: Optional[Union[str, Path]]`: 输出路径，保存评估结果。
  - `embedding_storage_mode: EmbeddingStorageMode`: 嵌入存储模式，默认为"none"。
  - `mini_batch_size: int`: 每个小批量的大小，默认为32。
  - `main_evaluation_metric: tuple[str, str]`: 主要评估指标，默认为("micro avg", "f1-score")。
  - `exclude_labels: Optional[list[str]]`: 排除的标签列表，默认为None。
  - `gold_label_dictionary: Optional[Dictionary]`: 真实标签字典，默认为None。
  - `return_loss: bool`: 是否返回损失，默认为True。
  - `**kwargs`: 其他可选参数。
- **返回值**: `Result`
  - **描述**: 返回评估结果，包括主要得分、详细结果、分类报告和分数。
- **范围**: 用于评估模型在给定数据点上的表现。

#### 方法：predict
- **函数名**: `predict`
- **参数**:
  - `sentences: Union[list[DT], DT]`: 输入的句子，可以是句子列表或单个句子。
  - `mini_batch_size: int`: 每个小批量的大小，默认为32。
  - `return_probabilities_for_all_classes: bool`: 是否返回所有类别的概率，默认为False。
  - `verbose: bool`: 是否输出详细信息，默认为False。
  - `label_name: Optional[str]`: 标签名称，默认为None。
  - `return_loss: bool`: 是否返回损失，默认为False。
  - `embedding_storage_mode: EmbeddingStorageMode`: 嵌入存储模式，默认为"none"。
- **返回值**: `Any`
  - **描述**: 返回预测结果，具体类型取决于实现。
- **范围**: 用于对输入句子进行预测。

#### 方法：_print_predictions
- **函数名**: `_print_predictions`
- **参数**:
  - `batch: list[DT]`: 输入的批量数据点。
  - `gold_label_type: str`: 真实标签的类型。
- **返回值**: `list[str]`
  - **描述**: 返回包含预测和真实标签的字符串列表。
- **范围**: 用于生成预测结果的打印信息。

#### 方法：get_used_tokens
- **函数名**: `get_used_tokens`
- **参数**:
  - `corpus: Corpus`: 输入的语料库。
  - `context_length: int`: 上下文长度，默认为0。
  - `respect_document_boundaries: bool`: 是否尊重文档边界，默认为True。
- **返回值**: `typing.Iterable[list[str]]`
  - **描述**: 返回用于模型的所有令牌的迭代器。
- **范围**: 用于获取模型使用的令牌。

#### 方法：load
- **函数名**: `load`
- **参数**:
  - `model_path: Union[str, Path, dict[str, Any]]`: 模型路径，可以是字符串、Path对象或字典。
- **返回值**: `Classifier`
  - **描述**: 返回加载的Classifier实例。
- **范围**: 用于加载已保存的模型。

## 示例调用

```python
from flair.data import Sentence
from flair.nn import Classifier

# 创建一个句子
sentence = Sentence('I love Berlin.')

# 加载NER标注器
tagger = Classifier.load('/mnt/autor_name/haoTingDeWenJianJia/flair/load_model_path/pytorch_model.bin')

# 对句子进行NER预测
tagger.predict(sentence)

# 打印带有所有注释的句子
print(sentence)
```