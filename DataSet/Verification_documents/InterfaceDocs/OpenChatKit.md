# API Documentation for ChatModel

## Class: ChatModel

The `ChatModel` class is designed to facilitate the interaction with a conversational AI model. It handles the loading of the model and tokenizer, as well as the generation of responses based on user prompts.

### Attributes:
- **human_id** (str): A string identifier for the human user, default is `"<human>"`.
- **bot_id** (str): A string identifier for the bot, default is `"<bot>"`.
- **_model** (transformers.AutoModelForCausalLM): The loaded conversational AI model.
- **_tokenizer** (transformers.AutoTokenizer): The tokenizer associated with the model.

### Method: `__init__`

```python
def __init__(self, model_name: str, gpu_id: int, max_memory: dict = None)
```

#### Parameters:
- **model_name** (str): The name or path of the pre-trained model to be loaded. This should correspond to a model available in the Hugging Face model hub.
- **gpu_id** (int): The ID of the GPU to which the model will be allocated. This should be a valid integer corresponding to the available GPUs on the machine.
- **max_memory** (dict, optional): A dictionary specifying the maximum memory allocation for each device. This is useful for loading large models on devices with limited VRAM. If `None`, the model will be loaded onto a single device.

#### Return Value:
- None

#### Purpose:
Initializes the `ChatModel` instance by loading the specified pre-trained model and tokenizer. It configures the model to run on the specified GPU and handles memory management based on the provided `max_memory` configuration.

---

### Method: `do_inference`

```python
def do_inference(self, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_k: int, stream_callback: callable = None) -> str
```

#### Parameters:
- **prompt** (str): The input text prompt that the model will respond to. This should be a string containing the conversation context or question.
- **max_new_tokens** (int): The maximum number of new tokens to generate in the response. This should be a positive integer.
- **do_sample** (bool): A flag indicating whether to use sampling for token generation. If `True`, the model will sample from the distribution of possible next tokens; if `False`, it will use greedy decoding.
- **temperature** (float): A value controlling the randomness of the sampling process. Higher values (e.g., >1.0) result in more random outputs, while lower values (e.g., <1.0) make the output more deterministic. Typical values range from 0.0 to 2.0.
- **top_k** (int): The number of highest probability vocabulary tokens to keep for top-k sampling. This should be a positive integer.
- **stream_callback** (callable, optional): A callback function that can be used to stream the output as it is generated. This is useful for real-time applications.

#### Return Value:
- **str**: The generated response from the model, which is a string containing the output text.

#### Purpose:
Generates a response based on the provided prompt using the loaded conversational AI model. It allows for customization of the generation process through parameters such as `max_new_tokens`, `do_sample`, `temperature`, and `top_k`. The method returns the model's output after removing the original prompt from the response.

