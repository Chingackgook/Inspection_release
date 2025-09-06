# API Documentation

## Functions

### `multinomial_sample_one_no_sync`
```python
def multinomial_sample_one_no_sync(probs_sort)
```
- **Parameters:**
  - `probs_sort` (torch.Tensor): A tensor containing sorted probabilities from which to sample.
  
- **Returns:**
  - torch.Tensor: The index of the sampled element.

- **Purpose:** 
  Performs multinomial sampling without CUDA synchronization, returning the index of the sampled element based on the provided probabilities.

---

### `logits_to_probs`
```python
def logits_to_probs(logits, previous_tokens: Optional[torch.Tensor] = None, temperature: torch.Tensor = 1.0, top_p: torch.Tensor = 1.0, repetition_penalty: torch.Tensor = 1.0) -> torch.Tensor
```
- **Parameters:**
  - `logits` (torch.Tensor): The raw output logits from the model.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `temperature` (torch.Tensor): Controls randomness in sampling (default is 1.0, range: [0, ∞)).
  - `top_p` (torch.Tensor): Cumulative probability threshold for top-p sampling (default is 1.0, range: [0, 1]).
  - `repetition_penalty` (torch.Tensor): Penalty for repeating tokens (default is 1.0, range: [0, 2)).
  
- **Returns:**
  - torch.Tensor: A tensor of probabilities derived from the logits.

- **Purpose:** 
  Converts logits to probabilities while applying temperature scaling, top-p sampling, and repetition penalties.

---

### `multinomial_sample_one_no_sync_agent`
```python
def multinomial_sample_one_no_sync_agent(probs_sort)
```
- **Parameters:**
  - `probs_sort` (torch.Tensor): A tensor containing sorted probabilities from which to sample.
  
- **Returns:**
  - torch.Tensor: The index of the sampled element.

- **Purpose:** 
  Similar to `multinomial_sample_one_no_sync`, this function performs multinomial sampling without CUDA synchronization for agent-specific use cases.

---

### `logits_to_probs_agent`
```python
def logits_to_probs_agent(logits, previous_tokens: Optional[torch.Tensor] = None, temperature: torch.Tensor = 1.0, top_p: torch.Tensor = 1.0, repetition_penalty: torch.Tensor = 1.0) -> torch.Tensor
```
- **Parameters:**
  - `logits` (torch.Tensor): The raw output logits from the model.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `temperature` (torch.Tensor): Controls randomness in sampling (default is 1.0, range: [0, ∞)).
  - `top_p` (torch.Tensor): Cumulative probability threshold for top-p sampling (default is 1.0, range: [0, 1]).
  - `repetition_penalty` (torch.Tensor): Penalty for repeating tokens (default is 1.0, range: [0, 2)).
  
- **Returns:**
  - torch.Tensor: A tensor of probabilities derived from the logits.

- **Purpose:** 
  Converts logits to probabilities for agent-specific use cases, applying temperature scaling, top-p sampling, and repetition penalties.

---

### `sample`
```python
def sample(logits, previous_tokens: Optional[torch.Tensor] = None, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]
```
- **Parameters:**
  - `logits` (torch.Tensor): The raw output logits from the model.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling (e.g., temperature, top_p, repetition_penalty).
  
- **Returns:**
  - Tuple[torch.Tensor, torch.Tensor]: The sampled token index and the corresponding probabilities.

- **Purpose:** 
  Samples a token from the logits using the specified sampling parameters and returns the sampled token along with its probabilities.

---

### `sample_agent`
```python
def sample_agent(logits, previous_tokens: Optional[torch.Tensor] = None, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]
```
- **Parameters:**
  - `logits` (torch.Tensor): The raw output logits from the model.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling (e.g., temperature, top_p, repetition_penalty).
  
- **Returns:**
  - Tuple[torch.Tensor, torch.Tensor]: The sampled token index and the corresponding probabilities.

- **Purpose:** 
  Similar to `sample`, this function samples a token from the logits for agent-specific use cases.

---

### `decode_one_token_ar_agent`
```python
def decode_one_token_ar_agent(model: DualARTransformer, x: torch.Tensor, input_pos: torch.Tensor, semantic_ids: list, previous_tokens: torch.Tensor = None, **sampling_kwargs) -> torch.Tensor
```
- **Parameters:**
  - `model` (DualARTransformer): The model used for decoding.
  - `x` (torch.Tensor): Input tensor for the model.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `semantic_ids` (list): List of semantic IDs.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The decoded token codes.

- **Purpose:** 
  Decodes a single token using the DualARTransformer model, applying the specified sampling parameters.

---

### `decode_one_token_naive_agent`
```python
def decode_one_token_naive_agent(model: NaiveTransformer, x: torch.Tensor, input_pos: torch.Tensor, semantic_ids: list, previous_tokens: torch.Tensor = None, **sampling_kwargs) -> torch.Tensor
```
- **Parameters:**
  - `model` (NaiveTransformer): The model used for decoding.
  - `x` (torch.Tensor): Input tensor for the model.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `semantic_ids` (list): List of semantic IDs.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The decoded token codes.

- **Purpose:** 
  Decodes a single token using the NaiveTransformer model, applying the specified sampling parameters.

---

### `decode_one_token_ar`
```python
def decode_one_token_ar(model: DualARTransformer, x: torch.Tensor, input_pos: torch.Tensor, semantic_ids: list, previous_tokens: torch.Tensor = None, **sampling_kwargs) -> torch.Tensor
```
- **Parameters:**
  - `model` (DualARTransformer): The model used for decoding.
  - `x` (torch.Tensor): Input tensor for the model.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `semantic_ids` (list): List of semantic IDs.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The decoded token codes.

- **Purpose:** 
  Decodes a single token using the DualARTransformer model, applying the specified sampling parameters.

---

### `decode_one_token_naive`
```python
def decode_one_token_naive(model: NaiveTransformer, x: torch.Tensor, input_pos: torch.Tensor, previous_tokens: torch.Tensor = None, **sampling_kwargs) -> torch.Tensor
```
- **Parameters:**
  - `model` (NaiveTransformer): The model used for decoding.
  - `x` (torch.Tensor): Input tensor for the model.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `previous_tokens` (Optional[torch.Tensor]): Tokens generated in the previous step (default is None).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The decoded token codes.

- **Purpose:** 
  Decodes a single token using the NaiveTransformer model, applying the specified sampling parameters.

---

### `decode_n_tokens`
```python
def decode_n_tokens(model: NaiveTransformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, semantic_ids: list, decode_one_token=decode_one_token_naive, **sampling_kwargs)
```
- **Parameters:**
  - `model` (NaiveTransformer): The model used for decoding.
  - `cur_token` (torch.Tensor): Current token tensor.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `num_new_tokens` (int): Number of new tokens to decode.
  - `semantic_ids` (list): List of semantic IDs.
  - `decode_one_token` (callable): Function to decode a single token (default is `decode_one_token_naive`).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The tensor of previous tokens.

- **Purpose:** 
  Decodes a specified number of new tokens using the provided model and decoding function.

---

### `generate`
```python
def generate(*, model: NaiveTransformer, prompt: torch.Tensor, max_new_tokens: int, decode_one_token=decode_one_token_naive, **sampling_kwargs) -> torch.Tensor
```
- **Parameters:**
  - `model` (NaiveTransformer): The model used for generation.
  - `prompt` (torch.Tensor): The input prompt tensor.
  - `max_new_tokens` (int): Maximum number of new tokens to generate.
  - `decode_one_token` (callable): Function to decode a single token (default is `decode_one_token_naive`).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - torch.Tensor: The generated token sequence.

- **Purpose:** 
  Generates a sequence of tokens based on the input prompt and specified parameters.

---

### `decode_n_tokens_agent`
```python
def decode_n_tokens_agent(model: NaiveTransformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, semantic_ids: list, im_end_id: int = 4, decode_one_token=decode_one_token_naive_agent, early_stop_threshold: float = 0.6, **sampling_kwargs)
```
- **Parameters:**
  - `model` (NaiveTransformer): The model used for decoding.
  - `cur_token` (torch.Tensor): Current token tensor.
  - `input_pos` (torch.Tensor): Position tensor for the input.
  - `num_new_tokens` (int): Number of new tokens to decode.
  - `semantic_ids` (list): List of semantic IDs.
  - `im_end_id` (int): ID for the end token (default is 4).
  - `decode_one_token` (callable): Function to decode a single token (default is `decode_one_token_naive_agent`).
  - `early_stop_threshold` (float): Threshold for early stopping (default is 0.6, range: [0, 1]).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - Generator: Yields the current token tensor.

- **Purpose:** 
  Decodes a specified number of new tokens for agent-specific use cases, yielding the current token tensor at each step.

---

### `generate_agent`
```python
def generate_agent(*, model: BaseTransformer, prompt: torch.Tensor, max_new_tokens: int, semantic_ids: list, im_end_id: int = 4, decode_one_token=decode_one_token_naive_agent, num_samples: int = 1, early_stop_threshold: float = 0.6, **sampling_kwargs)
```
- **Parameters:**
  - `model` (BaseTransformer): The model used for generation.
  - `prompt` (torch.Tensor): The input prompt tensor.
  - `max_new_tokens` (int): Maximum number of new tokens to generate.
  - `semantic_ids` (list): List of semantic IDs.
  - `im_end_id` (int): ID for the end token (default is 4).
  - `decode_one_token` (callable): Function to decode a single token (default is `decode_one_token_naive_agent`).
  - `num_samples` (int): Number of samples to generate (default is 1).
  - `early_stop_threshold` (float): Threshold for early stopping (default is 0.6, range: [0, 1]).
  - `**sampling_kwargs`: Additional keyword arguments for sampling.
  
- **Returns:**
  - Generator: Yields the generated token tensor.

- **Purpose:** 
  Generates a sequence of tokens based on the input prompt for agent-specific use cases, yielding the generated tokens at each step.

---

### `encode_tokens`
```python
def encode_tokens(tokenizer, string, device="cuda", prompt_tokens=None, num_codebooks=4)
```
- **Parameters:**
  - `tokenizer`: The tokenizer used for encoding.
  - `string` (str): The input string to encode.
  - `device` (str): The device to use for encoding (default is "cuda").
  - `prompt_tokens` (Optional[torch.Tensor]): Tokens from a previous prompt (default is None).
  - `num_codebooks` (int): Number of codebooks to use (default is 4).
  
- **Returns:**
  - torch.Tensor: The encoded tensor representation of the input string.

- **Purpose:** 
  Encodes a given string into a tensor representation using the specified tokenizer and parameters.

---

### `load_model`
```python
def load_model(checkpoint_path, device, precision, compile=False, is_agent=False)
```
- **Parameters:**
  - `checkpoint_path` (str): Path to the model checkpoint.
  - `device` (str | torch.device): Device to load the model onto.
  - `precision`: Precision for model weights.
  - `compile` (bool): Whether to compile the model (default is False).
  - `is_agent` (bool): Whether the model is for agent use (default is False).
  
- **Returns:**
  - Tuple[Union[NaiveTransformer, DualARTransformer], callable]: The loaded model and the decoding function.

- **Purpose:** 
  Loads a model from a checkpoint and prepares it for inference, returning the model and the decoding function.

---

## Classes

### `GenerateResponse`
```python
@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None
```
- **Attributes:**
  - `action` (Literal["sample", "next"]): Indicates the action type (either "sample" or "next").
  - `codes` (Optional[torch.Tensor]): The generated token codes (default is None).
  - `text` (Optional[str]): The text associated with the generated tokens (default is None).

- **Purpose:** 
  Represents the response from a generation process, encapsulating the action type, generated codes, and associated text.

---

### `generate_long`
```python
def generate_long(*, model, device: str | torch.device, decode_one_token: callable, text: str, num_samples: int = 1, max_new_tokens: int = 0, top_p: int = 0.7, repetition_penalty: float = 1.5, temperature: float = 0.7, compile: bool = False, iterative_prompt: bool = True, max_length: int = 2048, chunk_length: int = 150, prompt_text: Optional[str | list[str]] = None, prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None)
```
- **Parameters:**
  - `model`: The model used for generation.
  - `device` (str | torch.device): Device to run the model on.
  - `decode_one_token` (callable): Function to decode a single token.
  - `text` (str): The input text to generate from.
  - `num_samples` (int): Number of samples to generate (default is 1).
  - `max_new_tokens` (int): Maximum number of new tokens to generate (default is 0).
  - `top_p` (int): Cumulative probability threshold for top-p sampling (default is 0.7, range: (0, 1]).
  - `repetition_penalty` (float): Penalty for repeating tokens (default is 1.5, range: (0, 2)).
  - `temperature` (float): Controls randomness in sampling (default is 0.7, range: (0, 2)).
  - `compile` (bool): Whether to compile the model (default is False).
  - `iterative_prompt` (bool): Whether to use iterative prompting (default is True).
  - `max_length` (int): Maximum length of the generated sequence (default is 2048).
  - `chunk_length` (int): Length of each chunk for iterative prompting (default is 150).
  - `prompt_text` (Optional[str | list[str]]): Text prompts for generation (default is None).
  - `prompt_tokens` (Optional[torch.Tensor | list[torch.Tensor]]): Token prompts for generation (default is None).
  
- **Returns:**
  - Generator: Yields `GenerateResponse` objects containing generated codes and text.

- **Purpose:** 
  Generates long sequences of text based on the provided input text and parameters, yielding responses for each generated segment.

---

### `WrappedGenerateResponse`
```python
@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None
```
- **Attributes:**
  - `status` (Literal["success", "error"]): Indicates the status of the response (either "success" or "error").
  - `response` (Optional[GenerateResponse | Exception]): The response object or an exception (default is None).

- **Purpose:** 
  Wraps the response from a generation process, providing a status and the associated response or error.

---

### `GenerateRequest`
```python
@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue
```
- **Attributes:**
  - `request` (dict): The request parameters for generation.
  - `response_queue` (queue.Queue): The queue to which responses will be sent.

- **Purpose:** 
  Represents a request for generation, encapsulating the request parameters and the response queue for handling results.

---

### `launch_thread_safe_queue`
```python
def launch_thread_safe_queue(checkpoint_path, device, precision, compile: bool = False)
```
- **Parameters:**
  - `checkpoint_path` (str): Path to the model checkpoint.
  - `device` (str | torch.device): Device to load the model onto.
  - `precision`: Precision for model weights.
  - `compile` (bool): Whether to compile the model (default is False).
  
- **Returns:**
  - queue.Queue: The input queue for requests.

- **Purpose:** 
  Launches a thread-safe queue for handling generation requests, loading the model and preparing it for inference.

---

### `launch_thread_safe_queue_agent`
```python
def launch_thread_safe_queue_agent(checkpoint_path, device, precision, compile: bool = False)
```
- **Parameters:**
  - `checkpoint_path` (str): Path to the model checkpoint.
  - `device` (str | torch.device): Device to load the model onto.
  - `precision`: Precision for model weights.
  - `compile` (bool): Whether to compile the model (default is False).
  
- **Returns:**
  - Tuple[queue.Queue, AutoTokenizer, BaseModelArgs]: The input queue for requests, the tokenizer, and the model configuration.

- **Purpose:** 
  Launches a thread-safe queue for handling generation requests in agent-specific use cases, loading the model and preparing it for inference.