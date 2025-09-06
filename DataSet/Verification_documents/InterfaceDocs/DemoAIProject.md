# API Documentation for AIAgent Class

## AIAgent Class

### Description
The `AIAgent` class is an intelligent agent designed for interacting with the GPT API. It facilitates chat interactions, text generation, and text analysis using specified models.

### Attributes
- **api_key**: (str) The OpenAI API key used for authentication. This is retrieved from the environment variable `OPENAI_API_KEY`.
- **base_url**: (str) The base URL for the OpenAI API. This is retrieved from the environment variable `OPENAI_BASE_URL`.
- **model**: (str) The model name to be used for API calls. Defaults to the value of the environment variable `OPENAI_MODEL`.
- **headers**: (dict) The headers for API requests, including authorization and content type.

### Methods

#### __init__(self, model=None)
- **Parameters**:
  - `model` (Optional[str]): The name of the model to use. If not provided, it defaults to the value of the environment variable `OPENAI_MODEL`.
- **Returns**: None
- **Description**: Initializes the `AIAgent` instance, setting up the API key, base URL, model, and headers for API requests. Raises a `ValueError` if the API key is not set.

---

#### chat(self, message: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> Dict[str, Any]
- **Parameters**:
  - `message` (str): The user message to send to the AI.
  - `system_prompt` (Optional[str]): An optional system prompt to guide the AI's response.
  - `temperature` (float): A parameter to control the randomness of the AI's response. Range: [0.0, 1.0]. Default is 0.7.
- **Returns**: Dict[str, Any]
  - `success` (bool): Indicates whether the request was successful.
  - `reply` (Optional[str]): The AI's response if successful; otherwise, None.
  - `tokens_used` (dict): Metadata about token usage.
  - `error` (Optional[str]): Error message if the request failed.
- **Description**: Sends a chat message to the AI and retrieves the AI's response. Handles various exceptions and returns appropriate error messages.

---

#### generate_text(self, prompt: str) -> Dict[str, Any]
- **Parameters**:
  - `prompt` (str): The prompt text for which to generate a response.
- **Returns**: Dict[str, Any]
  - Same structure as the `chat` method.
- **Description**: Generates text based on the provided prompt using the AI. This method internally calls the `chat` method with a default temperature of 0.8.

---

#### analyze_text(self, text: str, analysis_type: str = "general") -> Dict[str, Any]
- **Parameters**:
  - `text` (str): The text to analyze.
  - `analysis_type` (str): The type of analysis to perform. Options include "general", "sentiment", and "summary". Default is "general".
- **Returns**: Dict[str, Any]
  - Same structure as the `chat` method.
- **Description**: Analyzes the provided text based on the specified analysis type. Uses a corresponding system prompt to guide the analysis.

---

#### set_model(self, model: str)
- **Parameters**:
  - `model` (str): The name of the model to set for future API calls.
- **Returns**: None
- **Description**: Updates the model used by the `AIAgent` instance.

---

#### get_model_info(self) -> Dict[str, str]
- **Returns**: Dict[str, str]
  - `current_model` (str): The name of the currently set model.
  - `base_url` (str): The base URL for the API.
- **Description**: Retrieves information about the current model and the base URL used for API requests.

---

This documentation provides a comprehensive overview of the `AIAgent` class, its methods, parameters, and return values, enabling users to effectively utilize the class for interacting with the GPT API.

