# API Documentation for MemvidChat

## Class: MemvidChat

### Description
`MemvidChat` is an enhanced conversational interface that supports multiple LLM (Large Language Model) providers. It allows users to interact with a knowledge base stored in video format, retrieving relevant context and generating responses based on user queries.

### Attributes
- **video_file**: `str`
  - Path to the video memory file.
  
- **index_file**: `str`
  - Path to the index JSON file.
  
- **llm_provider**: `str`
  - LLM provider (e.g., 'openai', 'google', 'anthropic').
  
- **llm_model**: `Optional[str]`
  - Model name (uses provider defaults if None).
  
- **llm_api_key**: `Optional[str]`
  - API key (uses environment variables if None).
  
- **config**: `Dict`
  - Configuration dictionary.
  
- **retriever_kwargs**: `Dict`
  - Additional arguments for `MemvidRetriever`.
  
- **retriever**: `MemvidRetriever`
  - Instance of the retriever for fetching context.
  
- **llm_client**: `LLMClient`
  - Instance of the LLM client for generating responses.
  
- **context_chunks**: `int`
  - Number of context chunks to retrieve for each query.
  
- **max_history**: `int`
  - Maximum number of conversation history entries to retain.
  
- **conversation_history**: `List[Dict[str, str]]`
  - List of conversation history entries.
  
- **session_id**: `Optional[str]`
  - Unique identifier for the chat session.
  
- **system_prompt**: `Optional[str]`
  - System prompt for the LLM.

### Method: `__init__`
```python
def __init__(self, video_file: str, index_file: str, llm_provider: str = 'google', llm_model: str = None, llm_api_key: str = None, config: Optional[Dict] = None, retriever_kwargs: Dict = None)
```
#### Parameters
- **video_file**: `str`
  - Path to the video memory file.
  
- **index_file**: `str`
  - Path to the index JSON file.
  
- **llm_provider**: `str`, optional (default: 'google')
  - LLM provider (e.g., 'openai', 'google', 'anthropic').
  
- **llm_model**: `Optional[str]`, optional
  - Model name (uses provider defaults if None).
  
- **llm_api_key**: `Optional[str]`, optional
  - API key (uses environment variables if None).
  
- **config**: `Optional[Dict]`, optional
  - Configuration dictionary.
  
- **retriever_kwargs**: `Dict`, optional
  - Additional arguments for `MemvidRetriever`.

#### Returns
- None

#### Description
Initializes the `MemvidChat` instance with the specified video file, index file, LLM provider, and other optional parameters. Sets up the retriever and LLM client.

### Method: `start_session`
```python
def start_session(self, system_prompt: str = None, session_id: str = None)
```
#### Parameters
- **system_prompt**: `Optional[str]`, optional
  - Custom system prompt for the session.
  
- **session_id**: `Optional[str]`, optional
  - Custom session identifier.

#### Returns
- None

#### Description
Starts a new chat session, initializing conversation history and setting the system prompt. If no session ID is provided, a timestamp-based ID is generated.

### Method: `chat`
```python
def chat(self, message: str, stream: bool = False, max_context_tokens: int = 2000) -> str
```
#### Parameters
- **message**: `str`
  - User message to send.
  
- **stream**: `bool`, optional (default: False)
  - Whether to stream the response.
  
- **max_context_tokens**: `int`, optional (default: 2000)
  - Maximum tokens to use for context.

#### Returns
- `str`
  - The response from the assistant.

#### Description
Sends a user message and retrieves a response using relevant context. If streaming is enabled, the response is streamed back to the user.

### Method: `interactive_chat`
```python
def interactive_chat(self)
```
#### Parameters
- None

#### Returns
- None

#### Description
Starts an interactive chat session, allowing the user to input messages and receive responses. Provides commands for clearing history, viewing stats, and exiting the session.

### Method: `search_context`
```python
def search_context(self, query: str, top_k: int = 5) -> List[str]
```
#### Parameters
- **query**: `str`
  - Search query to find relevant context.
  
- **top_k**: `int`, optional (default: 5)
  - Number of top results to return.

#### Returns
- `List[str]`
  - List of context strings retrieved based on the query.

#### Description
Searches for relevant context without generating a response. Returns a list of context strings based on the provided query.

### Method: `clear_history`
```python
def clear_history(self)
```
#### Parameters
- None

#### Returns
- None

#### Description
Clears the conversation history, resetting the chat session.

### Method: `export_conversation`
```python
def export_conversation(self, path: str)
```
#### Parameters
- **path**: `str`
  - Path to save the exported conversation history.

#### Returns
- None

#### Description
Exports the conversation history to a JSON file at the specified path, including session details and statistics.

### Method: `load_session`
```python
def load_session(self, session_file: str)
```
#### Parameters
- **session_file**: `str`
  - Path to the session file to load.

#### Returns
- None

#### Description
Loads a previous session from a file, restoring the conversation history and system prompt.

### Method: `reset_session`
```python
def reset_session(self)
```
#### Parameters
- None

#### Returns
- None

#### Description
Resets the conversation history and session ID, effectively starting a new session.

### Method: `get_stats`
```python
def get_stats(self) -> Dict
```
#### Parameters
- None

#### Returns
- `Dict`
  - A dictionary containing statistics about the current session.

#### Description
Retrieves statistics about the current chat session, including the number of messages exchanged and LLM provider details.

---

## Function: `chat_with_memory`
```python
def chat_with_memory(video_file: str, index_file: str, api_key: str = None, provider: str = 'google', model: str = None)
```
### Parameters
- **video_file**: `str`
  - Path to the video memory file.
  
- **index_file**: `str`
  - Path to the index file.
  
- **api_key**: `Optional[str]`, optional
  - LLM API key.
  
- **provider**: `str`, optional (default: 'google')
  - LLM provider (e.g., 'openai', 'google', 'anthropic').
  
- **model**: `Optional[str]`, optional
  - LLM model name.

### Returns
- None

### Description
Quick chat function for backwards compatibility. Initializes a `MemvidChat` instance and starts an interactive chat session.

---

## Function: `quick_chat`
```python
def quick_chat(video_file: str, index_file: str, message: str, provider: str = 'google', api_key: str = None) -> str
```
### Parameters
- **video_file**: `str`
  - Path to the video memory file.
  
- **index_file**: `str`
  - Path to the index file.
  
- **message**: `str`
  - Message to send to the assistant.
  
- **provider**: `str`, optional (default: 'google')
  - LLM provider (e.g., 'openai', 'google', 'anthropic').
  
- **api_key**: `Optional[str]`, optional
  - LLM API key.

### Returns
- `str`
  - Response from the assistant.

### Description
Quick single message chat function. Initializes a `MemvidChat` instance and sends a message, returning the assistant's response.

