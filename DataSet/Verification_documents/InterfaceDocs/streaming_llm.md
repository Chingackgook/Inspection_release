# API Documentation

## Function: `greedy_generate`

### Description
The `greedy_generate` function generates text based on a given input using a greedy search approach. It iteratively predicts the next token, updates the model's internal state, and prints the generated text until it reaches a maximum generation length or encounters an end-of-sequence token.

### Parameters
- `model` (torch.nn.Module): 
  - The model used for text generation. It should be a pre-trained language model compatible with the Hugging Face Transformers library.
  
- `tokenizer` (PreTrainedTokenizer):
  - The tokenizer corresponding to the model used. It is responsible for encoding input text and decoding output tokens.

- `input_ids` (torch.Tensor):
  - The input tensor containing the token IDs of the initial text. Shape: `(batch_size, sequence_length)`.

- `past_key_values` (tuple, optional):
  - A tuple containing the past key values for the attention mechanism. This is used for efficient decoding if the model supports caching. Use `None` if starting from scratch.

- `max_gen_len` (int):
  - The maximum number of tokens to generate. Must be a positive integer (range: `1` to `inf`).

### Returns
- `past_key_values` (tuple):
  - The updated past key values after generation, which can be used for subsequent decoding steps.

### Example
```python
# Example usage of greedy_generate
generated_past_key_values = greedy_generate(model, tokenizer, input_ids, None, max_gen_len=50)
```

### Notes
- The function suppresses warnings and uses PyTorch's no-gradient context to ensure efficient memory usage during text generation.
- The generated output is printed in real-time to the console, providing a streaming effect as the text is being generated.
- It stops generating tokens if it encounters the model's end-of-sequence token, which is defined by `tokenizer.eos_token_id`.

