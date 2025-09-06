# API Documentation

## Function: `search`

### Description
The `search` function performs a search operation based on the provided query text and other optional parameters. It retrieves relevant results from specified datasets, allowing for various types of searches.

### Parameters

- **query_text** (`str`): 
  - **Description**: The text to search for within the datasets.
  - **Value Range**: Any string value.
  
- **query_type** (`SearchType`, optional): 
  - **Description**: The type of search to perform. Defaults to `SearchType.GRAPH_COMPLETION`.
  - **Value Range**: Must be a valid `SearchType` enumeration value.

- **user** (`User`, optional): 
  - **Description**: The user performing the search. If not provided, a default user will be retrieved.
  - **Value Range**: Must be an instance of the `User` class or `None`.

- **datasets** (`Union[list[str], str]`, optional): 
  - **Description**: A list of dataset names or a single dataset name to search within. If provided as a string, it will be converted to a list.
  - **Value Range**: Can be a list of strings or a single string.

- **dataset_ids** (`Union[list[UUID], UUID]`, optional): 
  - **Description**: A list of dataset IDs or a single dataset ID to search within. This takes precedence over the `datasets` parameter.
  - **Value Range**: Can be a list of UUIDs or a single UUID.

- **system_prompt_path** (`str`, optional): 
  - **Description**: The path to the system prompt file used for the search. Defaults to "answer_simple_question.txt".
  - **Value Range**: Any valid file path string.

- **top_k** (`int`, optional): 
  - **Description**: The number of top results to return from the search. Defaults to 10.
  - **Value Range**: Must be a positive integer.

- **node_type** (`Optional[Type]`, optional): 
  - **Description**: The type of node to filter the search results. Defaults to `None`.
  - **Value Range**: Must be a valid type or `None`.

- **node_name** (`Optional[List[str]]`, optional): 
  - **Description**: A list of node names to filter the search results. Defaults to `None`.
  - **Value Range**: Can be a list of strings or `None`.

### Returns
- **Type**: `list`
- **Description**: A list of filtered search results based on the provided query and parameters.

### Raises
- **DatasetNotFoundError**: If no datasets are found based on the provided `datasets` parameter.

### Example Usage
```python
results = await search(
    query_text="What is the capital of France?",
    query_type=SearchType.GRAPH_COMPLETION,
    user=my_user,
    datasets=["dataset1", "dataset2"],
    top_k=5
)
```

### Purpose
The `search` function is designed to facilitate searching through datasets for specific information based on user queries, providing flexibility in terms of the datasets and search parameters used.

# API Documentation

## Function: `add`

### Description
The `add` function is responsible for adding data to a specified dataset. It can handle various types of data inputs and integrates with a pipeline to process the data ingestion.

### Parameters

- **data** (`Union[BinaryIO, list[BinaryIO], str, list[str]]`): 
  - **Description**: The data to be added to the dataset. This can be a single file, a list of files, or a string representing the data.
  - **Value Range**: Can be a single `BinaryIO` object, a list of `BinaryIO` objects, a single string, or a list of strings.

- **dataset_name** (`str`, optional): 
  - **Description**: The name of the dataset to which the data will be added. Defaults to "main_dataset".
  - **Value Range**: Any valid string representing a dataset name.

- **user** (`User`, optional): 
  - **Description**: The user performing the data addition. If not provided, the function will use a default user.
  - **Value Range**: Must be an instance of the `User` class or `None`.

- **node_set** (`Optional[List[str]]`, optional): 
  - **Description**: A list of node names to associate with the data being added. Defaults to `None`.
  - **Value Range**: Can be a list of strings or `None`.

- **vector_db_config** (`dict`, optional): 
  - **Description**: Configuration settings for the vector database. Defaults to `None`.
  - **Value Range**: Must be a dictionary or `None`.

- **graph_db_config** (`dict`, optional): 
  - **Description**: Configuration settings for the graph database. Defaults to `None`.
  - **Value Range**: Must be a dictionary or `None`.

- **dataset_id** (`UUID`, optional): 
  - **Description**: The unique identifier of the dataset to which the data will be added. If provided, it takes precedence over `dataset_name`.
  - **Value Range**: Must be a valid UUID or `None`.

### Returns
- **Type**: `Any`
- **Description**: Returns information about the pipeline run, which includes details about the data ingestion process.

### Example Usage
```python
pipeline_info = await add(
    data="path/to/data/file.csv",
    dataset_name="my_dataset",
    user=my_user,
    node_set=["node1", "node2"],
    vector_db_config={"param1": "value1"},
    graph_db_config={"param2": "value2"}
)
```

### Purpose
The `add` function is designed to facilitate the addition of data to datasets, allowing for flexible input types and configurations. It integrates with a processing pipeline to ensure that the data is properly ingested and associated with the specified dataset.

