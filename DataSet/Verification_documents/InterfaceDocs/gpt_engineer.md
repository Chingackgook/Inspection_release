# API Documentation

## Class: AI

### Description
The `AI` class interfaces with language models for conversation management and message serialization. It provides methods to start and advance conversations, handle message serialization, and implement backoff strategies for rate limit errors when interacting with the OpenAI API.

### Attributes
- **temperature** (`float`): The temperature setting for the language model. Default is `0.1`.
- **azure_endpoint** (`str`): The endpoint URL for the Azure-hosted language model. Default is `None`.
- **model_name** (`str`): The name of the language model to use. Default is `"gpt-4-turbo"`.
- **streaming** (`bool`): A flag indicating whether to use streaming for the language model. Default is `True`.
- **vision** (`bool`): A flag indicating if the model supports vision capabilities. Default is `False`.
- **llm** (`BaseChatModel`): The language model instance for conversation management.
- **token_usage_log** (`TokenUsageLog`): A log for tracking token usage during conversations.

### Method: `__init__`

#### Description
Initializes the `AI` class with specified parameters.

#### Parameters
- **model_name** (`str`, optional): The name of the model to use. Default is `"gpt-4-turbo"`.
- **temperature** (`float`, optional): The temperature to use for the model. Default is `0.1`.
- **azure_endpoint** (`str`, optional): The endpoint URL for the Azure-hosted language model. Default is `None`.
- **streaming** (`bool`, optional): A flag indicating whether to use streaming for the language model. Default is `True`.
- **vision** (`bool`, optional): A flag indicating if the model supports vision capabilities. Default is `False`.

#### Returns
None

---

### Method: `start`

#### Description
Starts the conversation with a system message and a user message.

#### Parameters
- **system** (`str`): The content of the system message.
- **user** (`str`): The content of the user message.
- **step_name** (`str`): The name of the step.

#### Returns
- **List[Message]**: The list of messages in the conversation.

---

### Method: `next`

#### Description
Advances the conversation by sending message history to the language model and updating with the response.

#### Parameters
- **messages** (`List[Message]`): The list of messages in the conversation.
- **prompt** (`Optional[str]`, optional): The prompt to use. Default is `None`.
- **step_name** (`str`): The name of the step.

#### Returns
- **List[Message]**: The updated list of messages in the conversation.

---

### Method: `backoff_inference`

#### Description
Performs inference using the language model while implementing an exponential backoff strategy. This function will retry the inference in case of a rate limit error from the OpenAI API.

#### Parameters
- **messages** (`List[Message]`): A list of chat messages which will be passed to the language model for processing.

#### Returns
- **Any**: The output from the language model after processing the provided messages.

#### Raises
- **openai.error.RateLimitError**: If the number of retries exceeds the maximum or if the rate limit persists beyond the allotted time.

---

### Method: `serialize_messages`

#### Description
Serializes a list of messages to a JSON string.

#### Parameters
- **messages** (`List[Message]`): The list of messages to serialize.

#### Returns
- **str**: The serialized messages as a JSON string.

---

### Method: `deserialize_messages`

#### Description
Deserializes a JSON string to a list of messages.

#### Parameters
- **jsondictstr** (`str`): The JSON string to deserialize.

#### Returns
- **List[Message]**: The deserialized list of messages.