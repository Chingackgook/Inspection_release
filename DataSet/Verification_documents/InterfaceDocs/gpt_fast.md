# API Documentation

## Function: `generate`

### Description
The `generate` function takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested. It supports both standard and speculative decoding, allowing for flexible generation of text based on the provided model.

### Parameters
- **model** (`Transformer`): The main transformer model used for generating tokens.
  
- **prompt** (`torch.Tensor`): A tensor containing the initial sequence of tokens to condition the generation on. The shape should be `(batch_size, sequence_length)`.

- **max_new_tokens** (`int`): The maximum number of new tokens to generate. This value should be greater than 0.

- **batch_size** (`int`): The number of sequences to generate in parallel. This value should be greater than 0.

- **interactive** (`bool`): A flag indicating whether the generation is interactive. If `True`, the maximum sequence length is limited to 350 tokens.

- **draft_model** (`Transformer`): An optional draft model used for speculative decoding. If `None`, standard decoding is performed.

- **speculate_k** (`Optional[int]`, default=8): The number of speculative tokens to consider during decoding. This value should be non-negative.

- **callback** (`callable`, default=lambda x: x): A callback function that is called with each generated token. It can be used for logging or processing tokens as they are generated.

- **sampling_kwargs** (`**kwargs`): Additional keyword arguments that can be passed to the sampling methods used in the generation process.

### Returns
- **torch.Tensor**: A tensor of shape `(batch_size, T + max_new_tokens)` containing the generated sequences, where `T` is the length of the input prompt.

- **dict**: A dictionary containing generation statistics, specifically:
  - **accept_counts**: A list of counts indicating how many tokens were accepted at each speculative step.

### Value Ranges
- `max_new_tokens`: Must be greater than 0.
- `batch_size`: Must be greater than 0.
- `speculate_k`: Must be non-negative.

### Purpose
The `generate` function is designed to facilitate the generation of text sequences based on a given prompt using a transformer model. It allows for both standard and speculative decoding, providing flexibility in how tokens are generated and enabling interactive use cases.

