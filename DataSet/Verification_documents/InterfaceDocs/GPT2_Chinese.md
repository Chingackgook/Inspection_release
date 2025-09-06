# API Documentation

## Function: `top_k_top_p_filtering`

### Description
Filters a distribution of logits using top-k and/or nucleus (top-p) filtering. This function is used to restrict the number of tokens considered for sampling based on their probabilities.

### Parameters
- **logits** (`torch.Tensor`): A tensor of shape (vocabulary size) representing the logits distribution.
- **top_k** (`int`, optional): The number of top tokens to keep based on their probabilities. Default is `0`. Must be greater than or equal to `0`.
- **top_p** (`float`, optional): The cumulative probability threshold for nucleus filtering. Default is `0.0`. Must be in the range `[0.0, 1.0]`.
- **filter_value** (`float`, optional): The value to assign to filtered logits. Default is `-float('Inf')`.

### Returns
- **torch.Tensor**: The filtered logits tensor.

### Purpose
This function is used to apply filtering techniques to logits to control the diversity of generated text by limiting the number of tokens considered for sampling.

---

## Function: `sample_sequence`

### Description
Generates a sequence of tokens from a given context using a specified language model. This function implements a sampling strategy with temperature, top-k, and top-p filtering.

### Parameters
- **model** (`GPT2LMHeadModel`): The language model used for generating tokens.
- **context** (`list[int]`): A list of token IDs representing the initial context for generation.
- **length** (`int`): The number of tokens to generate.
- **n_ctx** (`int`): The maximum context length for the model.
- **tokenizer**: The tokenizer used to convert text to token IDs and vice versa.
- **temperature** (`float`, optional): Controls the randomness of predictions. Default is `1.0`. Must be greater than `0`.
- **top_k** (`int`, optional): The number of top tokens to keep based on their probabilities. Default is `30`. Must be greater than or equal to `0`.
- **top_p** (`float`, optional): The cumulative probability threshold for nucleus filtering. Default is `0.0`. Must be in the range `[0.0, 1.0]`.
- **repitition_penalty** (`float`, optional): Penalty for repeating tokens. Default is `1.0`. Must be greater than or equal to `1.0`.
- **device** (`str`, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). Default is `'cpu'`.

### Returns
- **list[int]**: A list of generated token IDs.

### Purpose
This function generates a sequence of tokens based on the provided context and specified parameters, allowing for controlled randomness and diversity in the output.

---

## Function: `fast_sample_sequence`

### Description
Generates a sequence of tokens from a given context using a faster sampling method. This function is optimized for performance by leveraging cached hidden states.

### Parameters
- **model** (`GPT2LMHeadModel`): The language model used for generating tokens.
- **context** (`list[int]`): A list of token IDs representing the initial context for generation.
- **length** (`int`): The number of tokens to generate.
- **temperature** (`float`, optional): Controls the randomness of predictions. Default is `1.0`. Must be greater than `0`.
- **top_k** (`int`, optional): The number of top tokens to keep based on their probabilities. Default is `30`. Must be greater than or equal to `0`.
- **top_p** (`float`, optional): The cumulative probability threshold for nucleus filtering. Default is `0.0`. Must be in the range `[0.0, 1.0]`.
- **device** (`str`, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). Default is `'cpu'`.

### Returns
- **list[int]**: A list of generated token IDs.

### Purpose
This function provides a faster alternative for generating sequences by utilizing cached hidden states, improving performance while generating text.

---

## Function: `generate`

### Description
Generates a sequence of tokens based on the provided context and specified parameters. This function allows the user to choose between standard and fast sampling methods.

### Parameters
- **n_ctx** (`int`): The maximum context length for the model.
- **model** (`GPT2LMHeadModel`): The language model used for generating tokens.
- **context** (`list[int]`): A list of token IDs representing the initial context for generation.
- **length** (`int`): The number of tokens to generate.
- **tokenizer**: The tokenizer used to convert text to token IDs and vice versa.
- **temperature** (`float`, optional): Controls the randomness of predictions. Default is `1.0`. Must be greater than `0`.
- **top_k** (`int`, optional): The number of top tokens to keep based on their probabilities. Default is `0`. Must be greater than or equal to `0`.
- **top_p** (`float`, optional): The cumulative probability threshold for nucleus filtering. Default is `0.0`. Must be in the range `[0.0, 1.0]`.
- **repitition_penalty** (`float`, optional): Penalty for repeating tokens. Default is `1.0`. Must be greater than or equal to `1.0`.
- **device** (`str`, optional): The device to run the model on (e.g., 'cpu' or 'cuda'). Default is `'cpu'`.
- **is_fast_pattern** (`bool`, optional): Flag to indicate whether to use the fast sampling method. Default is `False`.

### Returns
- **list[int]**: A list of generated token IDs.

### Purpose
This function serves as a unified interface for generating sequences, allowing users to select between different sampling strategies based on their performance needs.

