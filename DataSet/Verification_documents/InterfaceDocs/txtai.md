# API Documentation

## Class: Embeddings

The `Embeddings` class provides an interface for creating and managing embeddings indexes for semantic search. It transforms data into embedding vectors, allowing for the retrieval of semantically similar documents.

### Attributes:
- `config`: Configuration settings for the embeddings index.
- `reducer`: Dimensionality reduction model for word vectors.
- `model`: Dense vector model for transforming data into similarity vectors.
- `ann`: Approximate nearest neighbor index for efficient searching.
- `ids`: Index ids when content is disabled.
- `database`: Document database for storing and retrieving documents.
- `functions`: Resolvable functions for various operations.
- `graph`: Graph network for advanced querying.
- `scoring`: Sparse vector scoring model.
- `query`: Query model for searching.
- `archive`: Index archive for saving and loading indexes.
- `indexes`: Subindexes for managing multiple embeddings instances.
- `models`: Cache for sharing models between embeddings.

### Method: `__init__(self, config=None, models=None, **kwargs)`

#### Parameters:
- `config` (dict, optional): Configuration settings for the embeddings index.
- `models` (optional): Models cache for sharing between embeddings.
- `kwargs` (optional): Additional configuration as keyword arguments.

#### Returns:
- None

#### Description:
Initializes a new instance of the `Embeddings` class, setting up the necessary configurations and attributes for managing embeddings indexes.

---

### Method: `score(self, documents)`

#### Parameters:
- `documents` (iterable): An iterable of documents, which can be in the form of (id, data, tags), (id, data), or just data.

#### Returns:
- None

#### Description:
Builds a term weighting scoring index for word vector models. This method is only applicable for models that support term weighting.

---

### Method: `index(self, documents, reindex=False, checkpoint=None)`

#### Parameters:
- `documents` (iterable): An iterable of documents, which can be in the form of (id, data, tags), (id, data), or just data.
- `reindex` (bool, optional): Indicates if this is a reindex operation. Defaults to `False`.
- `checkpoint` (str, optional): Optional checkpoint directory for enabling indexing restart.

#### Returns:
- None

#### Description:
Builds an embeddings index from the provided documents. This method overwrites any existing index and can handle reindexing operations.

---

### Method: `search(self, query, limit=None, weights=None, index=None, parameters=None, graph=False)`

#### Parameters:
- `query` (str): The input query for searching similar documents.
- `limit` (int, optional): The maximum number of results to return. Defaults to `None`.
- `weights` (list, optional): Hybrid score weights for scoring, if applicable.
- `index` (str, optional): The name of the index to search, if applicable.
- `parameters` (dict, optional): A dictionary of named parameters to bind to placeholders in the query.
- `graph` (bool, optional): If `True`, returns graph results. Defaults to `False`.

#### Returns:
- list: A list of tuples (id, score) for index search, or a list of dictionaries for an index + database search, or graph results if `graph` is set to `True.

#### Description:
Finds documents that are most similar to the input query. The method can perform an index search, an index + database search, or a graph search based on the configuration and query provided.

