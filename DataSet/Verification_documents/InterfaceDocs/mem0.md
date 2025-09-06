# API Documentation

## Class: Memory

The `Memory` class implements a multi-modal memory management system for AI agents, supporting cross-modal information storage, semantic retrieval, dynamic updates, and associative reasoning.

### Attributes:
- `config`: Configuration settings for the memory system.
- `custom_fact_extraction_prompt`: Custom prompt for fact extraction.
- `custom_update_memory_prompt`: Custom prompt for updating memory.
- `embedding_model`: Model used for embedding messages.
- `vector_store`: Storage system for vectorized memories.
- `llm`: Language model for generating responses.
- `db`: Database manager for history tracking.
- `collection_name`: Name of the collection in the vector store.
- `api_version`: Version of the API being used.
- `enable_graph`: Flag indicating if graph storage is enabled.
- `graph`: Graph memory manager if enabled.

### Method: `__init__(self, config: MemoryConfig = MemoryConfig())`
#### Parameters:
- `config`: An instance of `MemoryConfig`. Default is a new instance of `MemoryConfig`.
  
#### Return Value:
- None

#### Description:
Initializes the `Memory` class with the provided configuration settings.

---

### Method: `from_config(cls, config_dict: Dict[str, Any])`
#### Parameters:
- `config_dict`: A dictionary containing configuration settings.

#### Return Value:
- An instance of `Memory`.

#### Description:
Creates a `Memory` instance from a configuration dictionary, processing and validating the configuration.

---

### Method: `add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None, filters=None, infer=True, memory_type=None, prompt=None)`
#### Parameters:
- `messages`: A string or list of dictionaries containing messages to store in memory.
- `user_id`: (Optional) ID of the user creating the memory.
- `agent_id`: (Optional) ID of the agent creating the memory.
- `run_id`: (Optional) ID of the run creating the memory.
- `metadata`: (Optional) Additional metadata to store with the memory.
- `filters`: (Optional) Filters to apply to the search.
- `infer`: (Optional) Boolean indicating whether to infer memories. Default is `True`.
- `memory_type`: (Optional) Type of memory to create. Default is `None`.
- `prompt`: (Optional) Custom prompt for memory creation.

#### Return Value:
- A dictionary containing the result of the memory addition operation, including affected memories and graph memories.

#### Description:
Creates a new memory entry based on the provided messages and metadata.

---

### Method: `get(self, memory_id)`
#### Parameters:
- `memory_id`: ID of the memory to retrieve.

#### Return Value:
- A dictionary containing the retrieved memory or `None` if not found.

#### Description:
Retrieves a memory by its ID.

---

### Method: `get_all(self, user_id=None, agent_id=None, run_id=None, limit=100)`
#### Parameters:
- `user_id`: (Optional) ID of the user to filter memories.
- `agent_id`: (Optional) ID of the agent to filter memories.
- `run_id`: (Optional) ID of the run to filter memories.
- `limit`: (Optional) Maximum number of memories to return. Default is `100`.

#### Return Value:
- A list of all memories or a dictionary containing results and relations if graph storage is enabled.

#### Description:
Lists all memories, optionally filtered by user, agent, or run ID.

---

### Method: `search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None)`
#### Parameters:
- `query`: The search query string.
- `user_id`: (Optional) ID of the user to filter memories.
- `agent_id`: (Optional) ID of the agent to filter memories.
- `run_id`: (Optional) ID of the run to filter memories.
- `limit`: (Optional) Maximum number of results to return. Default is `100`.
- `filters`: (Optional) Additional filters to apply to the search.

#### Return Value:
- A list of search results or a dictionary containing results and relations if graph storage is enabled.

#### Description:
Searches for memories matching the provided query.

---

### Method: `update(self, memory_id, data)`
#### Parameters:
- `memory_id`: ID of the memory to update.
- `data`: New data to update the memory with.

#### Return Value:
- A dictionary confirming the update operation.

#### Description:
Updates a memory by its ID with the provided data.

---

### Method: `delete(self, memory_id)`
#### Parameters:
- `memory_id`: ID of the memory to delete.

#### Return Value:
- A dictionary confirming the deletion operation.

#### Description:
Deletes a memory by its ID.

---

### Method: `delete_all(self, user_id=None, agent_id=None, run_id=None)`
#### Parameters:
- `user_id`: (Optional) ID of the user to delete memories for.
- `agent_id`: (Optional) ID of the agent to delete memories for.
- `run_id`: (Optional) ID of the run to delete memories for.

#### Return Value:
- A dictionary confirming the deletion of all specified memories.

#### Description:
Deletes all memories associated with the specified user, agent, or run ID.

---

### Method: `history(self, memory_id)`
#### Parameters:
- `memory_id`: ID of the memory to get history for.

#### Return Value:
- A list of changes for the specified memory.

#### Description:
Retrieves the history of changes for a memory by its ID.

---

### Method: `reset(self)`
#### Parameters:
- None

#### Return Value:
- None

#### Description:
Resets the memory store by deleting the vector store collection and resetting the database.

---

### Method: `chat(self, query)`
#### Parameters:
- `query`: The input query for the chat function.

#### Return Value:
- Not implemented.

#### Description:
This method is not yet implemented and raises a `NotImplementedError`.