# 接口文档

## 类：`ChatCompletionAgent`

### 初始化方法：`__init__`

#### 参数说明：
- `arguments` (KernelArguments | None): 代理的内核参数。调用方法的参数优先于此处提供的参数。
- `description` (str | None): 代理的描述。
- `function_choice_behavior` (FunctionChoiceBehavior | None): 函数选择行为，用于确定如何向模型广告插件。
- `id` (str | None): 代理的唯一标识符。如果未提供，将生成一个唯一的 GUID。
- `instructions` (str | None): 代理的指令。
- `kernel` (Kernel | None): 内核实例。如果同时提供内核和服务，如果它们共享相同的 service_id 或 ai_model_id，则服务将优先使用。
- `name` (str | None): 代理的名称。
- `plugins` (list[KernelPlugin | object] | dict[str, KernelPlugin | object] | None): 代理的插件。如果与内核一起包含插件，则内核中已存在的任何插件将被覆盖。
- `prompt_template_config` (PromptTemplateConfig | None): 代理的提示模板配置。
- `service` (ChatCompletionClientBase | None): 聊天完成服务实例。如果提供了具有相同 service_id 或 ai_model_id 的内核，则服务将优先使用。

#### 返回值说明：
无返回值。

### 属性说明：
- `function_choice_behavior`: 函数选择行为，类型为 `FunctionChoiceBehavior | None`。
- `channel_type`: 类变量，类型为 `type[AgentChannel] | None`，默认为 `ChatHistoryChannel`。
- `service`: 聊天完成服务，类型为 `ChatCompletionClientBase | None`。

### 方法：`configure_service`

#### 参数说明：
无参数。

#### 返回值说明：
- 返回类型: `ChatCompletionAgent`，返回当前实例。

### 方法：`create_channel`

#### 参数说明：
- `chat_history` (ChatHistory | None): 渠道的聊天历史。如果为 None，将创建一个新的 ChatHistory 实例。
- `thread_id` (str | None): 线程的 ID。如果为 None，将创建一个新的线程。

#### 返回值说明：
- 返回类型: `AgentChannel`，返回一个 `ChatHistoryChannel` 实例。

### 方法：`get_response`

#### 参数说明：
- `messages` (str | ChatMessageContent | list[str | ChatMessageContent] | None): 输入的聊天消息内容，可以是字符串、ChatMessageContent 或字符串/ChatMessageContent 的列表。
- `thread` (AgentThread | None): 用于代理调用的线程。
- `arguments` (KernelArguments | None): 内核参数。
- `kernel` (Kernel | None): 内核实例。
- `**kwargs` (Any): 其他关键字参数。

#### 返回值说明：
- 返回类型: `AgentResponseItem[ChatMessageContent]`，返回代理的响应。

### 方法：`invoke`

#### 参数说明：
- `messages` (str | ChatMessageContent | list[str | ChatMessageContent] | None): 输入的聊天消息内容，可以是字符串、ChatMessageContent 或字符串/ChatMessageContent 的列表。
- `thread` (AgentThread | None): 用于代理调用的线程。
- `on_intermediate_message` (Callable[[ChatMessageContent], Awaitable[None]] | None): 处理代理执行中间步骤的回调函数。
- `arguments` (KernelArguments | None): 内核参数。
- `kernel` (Kernel | None): 内核实例。
- `**kwargs` (Any): 其他关键字参数。

#### 返回值说明：
- 返回类型: `AsyncIterable[AgentResponseItem[ChatMessageContent]]`，返回代理响应的异步可迭代对象。

### 方法：`invoke_stream`

#### 参数说明：
- `messages` (str | ChatMessageContent | list[str | ChatMessageContent] | None): 输入的聊天消息内容，可以是字符串、ChatMessageContent 或字符串/ChatMessageContent 的列表。
- `thread` (AgentThread | None): 用于代理调用的线程。
- `on_intermediate_message` (Callable[[ChatMessageContent], Awaitable[None]] | None): 处理代理执行中间步骤的回调函数。
- `arguments` (KernelArguments | None): 内核参数。
- `kernel` (Kernel | None): 内核实例。
- `**kwargs` (Any): 其他关键字参数。

#### 返回值说明：
- 返回类型: `AsyncIterable[AgentResponseItem[StreamingChatMessageContent]]`，返回代理响应的异步可迭代对象。

### 方法：`_inner_invoke`

#### 参数说明：
- `thread` (ChatHistoryAgentThread): 用于代理调用的线程。
- `history` (ChatHistory): 聊天历史。
- `on_intermediate_message` (Callable[[ChatMessageContent], Awaitable[None]] | None): 处理代理执行中间步骤的回调函数。
- `arguments` (KernelArguments | None): 内核参数。
- `kernel` (Kernel | None): 内核实例。
- `**kwargs` (Any): 其他关键字参数。

#### 返回值说明：
- 返回类型: `AsyncIterable[ChatMessageContent]`，返回代理响应的异步可迭代对象。

### 方法：`_prepare_agent_chat_history`

#### 参数说明：
- `history` (ChatHistory): 输入的聊天历史。
- `kernel` (Kernel): 内核实例。
- `arguments` (KernelArguments): 内核参数。

#### 返回值说明：
- 返回类型: `ChatHistory`，返回准备好的代理聊天历史。

### 方法：`_get_chat_completion_service_and_settings`

#### 参数说明：
- `kernel` (Kernel): 内核实例。
- `arguments` (KernelArguments): 内核参数。

#### 返回值说明：
- 返回类型: `tuple[ChatCompletionClientBase, PromptExecutionSettings]`，返回聊天完成服务和设置。

### 方法：`_capture_mutated_messages`

#### 参数说明：
- `agent_chat_history` (ChatHistory): 代理聊天历史。
- `start` (int): 开始索引。
- `thread` (ChatHistoryAgentThread): 用于代理调用的线程。
- `on_intermediate_message` (Callable[[ChatMessageContent], Awaitable[None]] | None): 处理代理执行中间步骤的回调函数。

#### 返回值说明：
无返回值。

## 调用示例

```python
# 创建 ChatCompletionAgent 实例
agent = ChatCompletionAgent(
    arguments=None,
    description="A chat completion agent",
    function_choice_behavior=None,
    id=None,
    instructions="Please assist with the following queries.",
    kernel=None,
    name="ChatAgent",
    plugins=None,
    prompt_template_config=None,
    service=None
)

# 创建聊天频道
channel = await agent.create_channel()

# 获取响应
response = await agent.get_response(messages="Hello, how can I help you?")
print(response.message.content)

# 调用代理
async for item in agent.invoke(messages="What is the weather today?"):
    print(item.message.content)

# 流式调用代理
async for item in agent.invoke_stream(messages="Tell me a joke."):
    print(item.message.content)
```