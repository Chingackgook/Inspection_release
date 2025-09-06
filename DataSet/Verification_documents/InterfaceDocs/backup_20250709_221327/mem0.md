# Memory Class API Documentation

## Class: Memory

### Initialization

#### `__init__(self, config: MemoryConfig = MemoryConfig())`
- **Parameters:**
  - `config` (MemoryConfig): Configuration object for initializing the memory system. Defaults to a new instance of `MemoryConfig`.
  
- **Attributes:**
  - `config`: Stores the configuration for the memory system.
  - `custom_fact_extraction_prompt`: Custom prompt for fact extraction.
  - `custom_update_memory_prompt`: Custom prompt for updating memory.
  - `embedding_model`: Model used for embedding messages.
  - `vector_store`: Store for vector representations of memories.
  - `llm`: Language model for generating responses.
  - `db`: Database manager for history tracking.
  - `collection_name`: Name of the vector store collection.
  - `api_version`: Version of the API.
  - `enable_graph`: Flag indicating if graph storage is enabled.
  - `graph`: Graph memory manager if enabled.
  - `_telemetry_vector_store`: Telemetry vector store for tracking events.

### Methods

#### `from_config(cls, config_dict: Dict[str, Any])`
- **Parameters:**
  - `config_dict` (Dict[str, Any]): Dictionary containing configuration parameters.
  
- **Returns:**
  - `Memory`: An instance of the `Memory` class initialized with the provided configuration.

#### `add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None, filters=None, infer=True, memory_type=None, prompt=None)`
- **Parameters:**
  - `messages` (str or List[Dict[str, str]]): Messages to store in memory.
  - `user_id` (str, optional): ID of the user creating the memory. Defaults to None.
  - `agent_id` (str, optional): ID of the agent creating the memory. Defaults to None.
  - `run_id` (str, optional): ID of the run creating the memory. Defaults to None.
  - `metadata` (dict, optional): Metadata to store with the memory. Defaults to None.
  - `filters` (dict, optional): Filters to apply to the search. Defaults to None.
  - `infer` (bool, optional): Whether to infer the memories. Defaults to True.
  - `memory_type` (str, optional): Type of memory to create. Defaults to None.
  - `prompt` (str, optional): Prompt to use for memory creation. Defaults to None.
  
- **Returns:**
  - `dict`: A dictionary containing the result of the memory addition operation, including added, updated, and deleted memories.

#### `get(self, memory_id)`
- **Parameters:**
  - `memory_id` (str): ID of the memory to retrieve.
  
- **Returns:**
  - `dict`: Retrieved memory, including its metadata.

#### `get_all(self, user_id=None, agent_id=None, run_id=None, limit=100)`
- **Parameters:**
  - `user_id` (str, optional): ID of the user to filter memories. Defaults to None.
  - `agent_id` (str, optional): ID of the agent to filter memories. Defaults to None.
  - `run_id` (str, optional): ID of the run to filter memories. Defaults to None.
  - `limit` (int, optional): Limit the number of results. Defaults to 100.
  
- **Returns:**
  - `list`: List of all memories, optionally filtered by user, agent, or run ID.

#### `search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None)`
- **Parameters:**
  - `query` (str): Query to search for.
  - `user_id` (str, optional): ID of the user to filter memories. Defaults to None.
  - `agent_id` (str, optional): ID of the agent to filter memories. Defaults to None.
  - `run_id` (str, optional): ID of the run to filter memories. Defaults to None.
  - `limit` (int, optional): Limit the number of results. Defaults to 100.
  - `filters` (dict, optional): Filters to apply to the search. Defaults to None.
  
- **Returns:**
  - `list`: List of search results, including matched memories.

#### `update(self, memory_id, data)`
- **Parameters:**
  - `memory_id` (str): ID of the memory to update.
  - `data` (dict): Data to update the memory with.
  
- **Returns:**
  - `dict`: Confirmation message indicating the memory was updated successfully.

#### `delete(self, memory_id)`
- **Parameters:**
  - `memory_id` (str): ID of the memory to delete.
  
- **Returns:**
  - `dict`: Confirmation message indicating the memory was deleted successfully.

#### `delete_all(self, user_id=None, agent_id=None, run_id=None)`
- **Parameters:**
  - `user_id` (str, optional): ID of the user to delete memories for. Defaults to None.
  - `agent_id` (str, optional): ID of the agent to delete memories for. Defaults to None.
  - `run_id` (str, optional): ID of the run to delete memories for. Defaults to None.
  
- **Returns:**
  - `dict`: Confirmation message indicating all specified memories were deleted successfully.

#### `history(self, memory_id)`
- **Parameters:**
  - `memory_id` (str): ID of the memory to get history for.
  
- **Returns:**
  - `list`: List of changes for the specified memory.

#### `reset(self)`
- **Returns:**
  - `None`: Resets the memory store, deleting all memories and recreating the vector store.

### Example Usage

```python
# Initialize Memory with default configuration
memory_instance = Memory()

# Add a memory
result = memory_instance.add(messages="This is a test memory.", user_id="user123")

# Retrieve a memory by ID
retrieved_memory = memory_instance.get(memory_id=result['results'][0]['id'])

# Update a memory
update_result = memory_instance.update(memory_id=retrieved_memory['id'], data={"text": "Updated memory content."})

# Delete a memory
delete_result = memory_instance.delete(memory_id=retrieved_memory['id'])

# Get all memories
all_memories = memory_instance.get_all(user_id="user123")

# Search for memories
search_results = memory_instance.search(query="test", user_id="user123")

# Reset the memory store
memory_instance.reset()
``` 

This documentation provides a comprehensive overview of the `Memory` class, its initialization, attributes, and methods, along with example usage for clarity.