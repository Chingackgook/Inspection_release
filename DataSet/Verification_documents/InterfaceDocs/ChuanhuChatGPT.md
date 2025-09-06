# API Documentation

## Class: OpenAIVisionClient

### Description
The `OpenAIVisionClient` class is an interface for interacting with OpenAI's vision models. It extends the `BaseLLMModel` class and provides core functionalities for generating responses. The class supports two modes of response generation: obtaining a complete response at once and streaming partial responses iteratively.

### Attributes
- `model_name`: The name of the model being used.
- `api_key`: The API key for authentication with OpenAI services.
- `user_name`: The name of the user (optional).
- `chat_completion_url`: URL for chat completion requests.
- `images_completion_url`: URL for image completion requests.
- `openai_api_base`: Base URL for OpenAI API.
- `balance_api_url`: URL for checking account balance.
- `usage_api_url`: URL for checking API usage.
- `headers`: HTTP headers for API requests.
- `history`: List to store conversation history.
- `system_prompt`: System prompt for the model.
- `temperature`: Sampling temperature for response generation.
- `top_p`: Nucleus sampling parameter.
- `n_choices`: Number of response choices to generate.
- `max_generation_token`: Maximum number of tokens for generation.
- `presence_penalty`: Penalty for new tokens based on their presence in the text.
- `frequency_penalty`: Penalty for new tokens based on their frequency in the text.
- `stop_sequence`: Sequence to stop generation.
- `logit_bias`: Bias for specific tokens.
- `user_identifier`: Identifier for the user.
- `config`: Configuration settings for the model.
- `multimodal`: Indicates if the model supports multiple modalities.
- `description`: Description of the model.
- `placeholder`: Placeholder text for user input.
- `token_upper_limit`: Upper limit for token usage.
- `all_token_counts`: List to track token counts for each interaction.
- `chatbot`: List to manage chatbot interactions.

### Method: `__init__(model_name, api_key, user_name="")`
#### Parameters
- `model_name` (str): The name of the model to be used.
- `api_key` (str): The API key for authentication.
- `user_name` (str, optional): The name of the user (default is an empty string).

#### Return Value
None

#### Description
Initializes an instance of the `OpenAIVisionClient` class, setting up the model name, API key, and user name. It also configures the necessary URLs for API interactions.

---

### Method: `get_answer_stream_iter()`
#### Parameters
None

#### Return Value
Iterator yielding strings.

#### Description
Generates responses in a streaming manner, yielding partial responses as they are generated. This allows for real-time updates during the response generation process.

---

### Method: `get_answer_at_once()`
#### Parameters
None

#### Return Value
Tuple (str, int): The generated response and the total token count used.

#### Description
Obtains a complete response from the model in one request. Returns the response content and the total number of tokens used in the request.

---

### Method: `count_token(user_input)`
#### Parameters
- `user_input` (str): The input text for which to count tokens.

#### Return Value
int: The total number of tokens.

#### Description
Counts the number of tokens in the provided user input, considering the system prompt if applicable.

---

### Method: `count_image_tokens(width: int, height: int)`
#### Parameters
- `width` (int): The width of the image.
- `height` (int): The height of the image.

#### Return Value
int: The total number of tokens required for the image.

#### Description
Calculates the number of tokens needed for processing an image based on its dimensions.

---

### Method: `billing_info()`
#### Parameters
None

#### Return Value
str: A string containing billing information.

#### Description
Retrieves and returns the billing information for the current month, including total usage and percentage of the usage limit.

---

### Method: `set_key(new_access_key)`
#### Parameters
- `new_access_key` (str): The new API key to be set.

#### Return Value
None

#### Description
Sets a new API key for the client and refreshes the headers for subsequent requests.

---

### Method: `auto_name_chat_history(name_chat_method, user_question, single_turn_checkbox)`
#### Parameters
- `name_chat_method` (str): The method to use for naming the chat history.
- `user_question` (str): The user's question to base the name on.
- `single_turn_checkbox` (bool): Indicates if the chat is a single turn.

#### Return Value
None

#### Description
Automatically names the chat history based on the specified method, using the user's question or the assistant's answer.

---

### Method: `__init__(model_name, user="", config=None)`
#### Parameters
- `model_name` (str): The name of the model.
- `user` (str, optional): The name of the user (default is an empty string).
- `config` (dict, optional): Configuration settings for the model (default is None).

#### Return Value
None

#### Description
Initializes an instance of the `BaseLLMModel` class, setting up the model name, user, and configuration settings.

---

### Method: `predict(inputs, chatbot, use_websearch=False, files=None, reply_language="中文", should_check_token_count=True)`
#### Parameters
- `inputs` (str or list): The input text or list of inputs for the model.
- `chatbot` (list): The current state of the chatbot conversation.
- `use_websearch` (bool, optional): Indicates if web search should be used (default is False).
- `files` (optional): Files to be processed (default is None).
- `reply_language` (str, optional): Language for the reply (default is "中文").
- `should_check_token_count` (bool, optional): Indicates if token count should be checked (default is True).

#### Return Value
Iterator yielding tuples (list, str): The updated chatbot state and status text.

#### Description
Generates a response based on the provided inputs and updates the chatbot state. It can handle both single-turn and multi-turn conversations.

---

### Method: `retry(chatbot, use_websearch=False, files=None, reply_language="中文")`
#### Parameters
- `chatbot` (list): The current state of the chatbot conversation.
- `use_websearch` (bool, optional): Indicates if web search should be used (default is False).
- `files` (optional): Files to be processed (default is None).
- `reply_language` (str, optional): Language for the reply (default is "中文").

#### Return Value
Iterator yielding tuples (list, str): The updated chatbot state and status text.

#### Description
Retries the last interaction with the model, using the previous inputs to generate a new response. This is useful for recovering from errors or improving responses.