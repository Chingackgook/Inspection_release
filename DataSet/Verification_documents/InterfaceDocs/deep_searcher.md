# API Documentation

## Functions

### `query`

#### Description
Queries the knowledge base with a specified question and generates an answer based on the retrieved information.

#### Parameters
- **original_query** (str): The question or query to search for.
- **max_iter** (int, optional): Maximum number of iterations for the search process. Default is 3. Must be a positive integer.

#### Returns
- **Tuple[str, List[RetrievalResult], int]**: A tuple containing:
  - The generated answer as a string.
  - A list of retrieval results that were used to generate the answer.
  - The number of tokens consumed during the process.

---

### `retrieve`

#### Description
Retrieves relevant information from the knowledge base without generating an answer.

#### Parameters
- **original_query** (str): The question or query to search for.
- **max_iter** (int, optional): Maximum number of iterations for the search process. Default is 3. Must be a positive integer.

#### Returns
- **Tuple[List[RetrievalResult], List[str], int]**: A tuple containing:
  - A list of retrieval results.
  - An empty list (placeholder for future use).
  - The number of tokens consumed during the process.

---

### `naive_retrieve`

#### Description
Performs a simple retrieval from the knowledge base using the naive RAG approach.

#### Parameters
- **query** (str): The question or query to search for.
- **collection** (str, optional): The name of the collection to search in. If None, searches in all collections. Default is None.
- **top_k** (int, optional): The maximum number of results to return. Default is 10. Must be a positive integer.

#### Returns
- **List[RetrievalResult]**: A list of retrieval results.

---

### `naive_rag_query`

#### Description
Queries the knowledge base using the naive RAG approach and generates an answer based on the retrieved information.

#### Parameters
- **query** (str): The question or query to search for.
- **collection** (str, optional): The name of the collection to search in. If None, searches in all collections. Default is None.
- **top_k** (int, optional): The maximum number of results to consider. Default is 10. Must be a positive integer.

#### Returns
- **Tuple[str, List[RetrievalResult]]**: A tuple containing:
  - The generated answer as a string.
  - A list of retrieval results that were used to generate the answer.

--- 

This documentation provides a clear understanding of the purpose, parameters, and return values of each function, ensuring that users can effectively utilize the API.

