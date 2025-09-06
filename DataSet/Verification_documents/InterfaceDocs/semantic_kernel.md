# API Documentation

## Class: `ChatCompletionAgent`

### Description
`ChatCompletionAgent` is a class that defines a chat completion agent based on a chat completion service (such as OpenAI or Azure). It is designed to manage multi-turn conversations, handle function calls, and interact with large language models (LLMs).

### Attributes
- **function_choice_behavior**: `FunctionChoiceBehavior | None`
  - Default: `FunctionChoiceBehavior.Auto()`
  - Description: Determines how and which plugins are advertised to the model.

- **channel_type**: `ClassVar[type[AgentChannel] | None]`
  - Default: `ChatHistoryChannel`
  - Description: The type of channel used for managing chat history.

- **service**: `ChatCompletionClientBase | None`
  - Default: `None`
  - Description: The chat completion service instance used by the agent.

### Method: `__init__`
```python
def __init__(
    self,
    *,
    arguments: KernelArguments | None = None,
    description: str | None = None,
    function_choice_behavior: FunctionChoiceBehavior | None = None,
    id: str | None = None,
    instructions: str | None = None,
    kernel: "Kernel | None" = None,
    name: str | None = None,
    plugins: list[KernelPlugin | object] | dict[str, KernelPlugin | object] | None = None,
    prompt_template_config: PromptTemplateConfig | None = None,
    service: ChatCompletionClientBase | None = None,
) -> None:
```
#### Parameters
- **arguments**: (Optional) The kernel arguments for the agent. Overrides method arguments if provided.
- **description**: (Optional) A brief description of the agent.
- **function_choice_behavior**: (Optional) Specifies the function choice behavior for plugin advertisement.
- **id**: (Optional) A unique identifier for the agent. If not provided, a GUID will be generated.
- **instructions**: (Optional) Instructions for the agent.
- **kernel**: (Optional) The kernel instance. If both a kernel and a service are provided, the service takes precedence if they share the same service_id or ai_model_id.
- **name**: (Optional) The name of the agent.
- **plugins**: (Optional) A list or dictionary of plugins for the agent. Existing plugins in the kernel will be overwritten if included.
- **prompt_template_config**: (Optional) Configuration for the prompt template used by the agent.
- **service**: (Optional) The chat completion service instance. Takes precedence over the kernel if they share the same service_id or ai_model_id.

#### Return Value
- None

#### Purpose
Initializes a new instance of `ChatCompletionAgent` with the specified parameters.

### Method: `configure_service`
```python
@model_validator(mode="after")
def configure_service(self) -> "ChatCompletionAgent":
```
#### Parameters
- None

#### Return Value
- `ChatCompletionAgent`: The configured instance of the agent.

#### Purpose
Configures the service used by the `ChatCompletionAgent`. Validates that the service is an instance of `ChatCompletionClientBase` and adds it to the kernel.

### Method: `create_channel`
```python
async def create_channel(
    self, chat_history: ChatHistory | None = None, thread_id: str | None = None
) -> AgentChannel:
```
#### Parameters
- **chat_history**: (Optional) The chat history for the channel. If `None`, a new `ChatHistory` instance will be created.
- **thread_id**: (Optional) The ID of the thread. If `None`, a new thread will be created.

#### Return Value
- `AgentChannel`: An instance of `AgentChannel` containing the chat history.

#### Purpose
Creates a `ChatHistoryChannel` for managing chat history, either using existing chat history or creating a new one.

### Method: `get_response`
```python
@trace_agent_get_response
@override
async def get_response(
    self,
    *,
    messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
    thread: AgentThread | None = None,
    arguments: KernelArguments | None = None,
    kernel: "Kernel | None" = None,
    **kwargs: Any,
) -> AgentResponseItem[ChatMessageContent]:
```
#### Parameters
- **messages**: (Optional) The input chat message content, which can be a string, `ChatMessageContent`, or a list of either.
- **thread**: (Optional) The thread to use for agent invocation.
- **arguments**: (Optional) The kernel arguments for the invocation.
- **kernel**: (Optional) The kernel instance to use.
- **kwargs**: Additional keyword arguments.

#### Return Value
- `AgentResponseItem[ChatMessageContent]`: An item containing the response message from the agent.

#### Purpose
Retrieves a response from the agent based on the provided messages and thread.

### Method: `invoke`
```python
@trace_agent_invocation
@override
async def invoke(
    self,
    *,
    messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
    thread: AgentThread | None = None,
    on_intermediate_message: Callable[[ChatMessageContent], Awaitable[None]] | None = None,
    arguments: KernelArguments | None = None,
    kernel: "Kernel | None" = None,
    **kwargs: Any,
) -> AsyncIterable[AgentResponseItem[ChatMessageContent]]:
```
#### Parameters
- **messages**: (Optional) The input chat message content, which can be a string, `ChatMessageContent`, or a list of either.
- **thread**: (Optional) The thread to use for agent invocation.
- **on_intermediate_message**: (Optional) A callback function to handle intermediate messages during execution.
- **arguments**: (Optional) The kernel arguments for the invocation.
- **kernel**: (Optional) The kernel instance to use.
- **kwargs**: Additional keyword arguments.

#### Return Value
- `AsyncIterable[AgentResponseItem[ChatMessageContent]]`: An asynchronous iterable of response items from the agent.

#### Purpose
Invokes the chat history handler and yields responses from the agent based on the provided messages.

### Method: `invoke_stream`
```python
@trace_agent_invocation
@override
async def invoke_stream(
    self,
    *,
    messages: str | ChatMessageContent | list[str | ChatMessageContent] | None = None,
    thread: AgentThread | None = None,
    on_intermediate_message: Callable[[ChatMessageContent], Awaitable[None]] | None = None,
    arguments: KernelArguments | None = None,
    kernel: "Kernel | None" = None,
    **kwargs: Any,
) -> AsyncIterable[AgentResponseItem[StreamingChatMessageContent]]:
```
#### Parameters
- **messages**: (Optional) The input chat message content, which can be a string, `ChatMessageContent`, or a list of either.
- **thread**: (Optional) The thread to use for agent invocation.
- **on_intermediate_message**: (Optional) A callback function to handle intermediate messages during execution.
- **arguments**: (Optional) The kernel arguments for the invocation.
- **kernel**: (Optional) The kernel instance to use.
- **kwargs**: Additional keyword arguments.

#### Return Value
- `AsyncIterable[AgentResponseItem[StreamingChatMessageContent]]`: An asynchronous iterable of streaming response items from the agent.

#### Purpose
Invokes the chat history handler in streaming mode, yielding responses as they are generated.