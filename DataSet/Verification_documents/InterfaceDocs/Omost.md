# API Documentation for StableDiffusionXLOmostPipeline

## Class: StableDiffusionXLOmostPipeline

### Description
The `StableDiffusionXLOmostPipeline` class is an implementation of a pipeline for image generation using the Stable Diffusion XL model. It extends the `StableDiffusionXLImg2ImgPipeline` and provides additional functionalities for encoding prompts and generating images based on those prompts.

### Attributes
- `k_model`: An instance of `KModel` initialized with the UNet model.
- `unet`: The UNet model used for image generation.
- `text_encoder`: The text encoder for processing text prompts.
- `text_encoder_2`: A second text encoder for processing additional text prompts.
- `tokenizer`: The tokenizer associated with the first text encoder.
- `tokenizer_2`: The tokenizer associated with the second text encoder.

### Method: `__init__`
```python
def __init__(self, *args, **kwargs)
```
#### Description
Initializes the `StableDiffusionXLOmostPipeline` class, setting up the UNet model and attention processors.

#### Parameters
- `*args`: Variable length argument list.
- `**kwargs`: Arbitrary keyword arguments.

#### Return Value
None

---

### Method: `encode_bag_of_subprompts_greedy`
```python
def encode_bag_of_subprompts_greedy(self, prefixes: list[str], suffixes: list[str])
```
#### Description
Encodes a list of prefixes and suffixes into a format suitable for the model, using a greedy partitioning strategy to manage token limits.

#### Parameters
- `prefixes` (list[str]): A list of prefix strings to be encoded.
- `suffixes` (list[str]): A list of suffix strings to be encoded.

#### Return Value
- `dict`: A dictionary containing:
  - `cond` (torch.FloatTensor): The concatenated encoded representations of the prefixes and suffixes.
  - `pooler` (torch.FloatTensor): The pooled representation of the encoded prompts.

---

### Method: `all_conds_from_canvas`
```python
def all_conds_from_canvas(self, canvas_outputs, negative_prompt)
```
#### Description
Processes canvas outputs to extract positive and negative conditions for image generation, encoding the prompts accordingly.

#### Parameters
- `canvas_outputs`: A dictionary containing outputs from the canvas, including masks and conditions.
- `negative_prompt` (str): A negative prompt string to be encoded.

#### Return Value
- `tuple`: A tuple containing:
  - `positive_result` (list): A list of tuples with masks and positive conditions.
  - `positive_pooler` (torch.FloatTensor): The pooled representation of positive conditions.
  - `negative_result` (list): A list of tuples with masks and negative conditions.
  - `negative_pooler` (torch.FloatTensor): The pooled representation of negative conditions.

---

### Method: `encode_cropped_prompt_77tokens`
```python
def encode_cropped_prompt_77tokens(self, prompt: str)
```
#### Description
Encodes a single prompt string into a fixed-length representation suitable for the model, ensuring it adheres to token limits.

#### Parameters
- `prompt` (str): The prompt string to be encoded.

#### Return Value
- `tuple`: A tuple containing:
  - `prompt_embeds` (torch.FloatTensor): The encoded representation of the prompt.
  - `pooled_prompt_embeds` (torch.FloatTensor): The pooled representation of the encoded prompt.

---

### Method: `__call__`
```python
def __call__(self, initial_latent: torch.FloatTensor = None, strength: float = 1.0, num_inference_steps: int = 25, guidance_scale: float = 5.0, batch_size: Optional[int] = 1, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, prompt_embeds: Optional[torch.FloatTensor] = None, negative_prompt_embeds: Optional[torch.FloatTensor] = None, pooled_prompt_embeds: Optional[torch.FloatTensor] = None, negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None, cross_attention_kwargs: Optional[dict] = None)
```
#### Description
Generates images based on the provided latent inputs and prompt embeddings, performing inference through the model.

#### Parameters
- `initial_latent` (torch.FloatTensor, optional): The initial latent tensor to start the generation process.
- `strength` (float, optional): The strength of the transformation (default: 1.0). Range: [0.0, 1.0].
- `num_inference_steps` (int, optional): The number of inference steps to perform (default: 25). Range: [1, ∞).
- `guidance_scale` (float, optional): The scale for guidance during generation (default: 5.0). Range: [0.0, ∞).
- `batch_size` (int, optional): The number of images to generate in a single batch (default: 1). Range: [1, ∞).
- `generator` (Union[torch.Generator, List[torch.Generator]], optional): A random number generator for reproducibility.
- `prompt_embeds` (torch.FloatTensor, optional): Pre-encoded prompt embeddings.
- `negative_prompt_embeds` (torch.FloatTensor, optional): Pre-encoded negative prompt embeddings.
- `pooled_prompt_embeds` (torch.FloatTensor, optional): Pooled embeddings for the positive prompts.
- `negative_pooled_prompt_embeds` (torch.FloatTensor, optional): Pooled embeddings for the negative prompts.
- `cross_attention_kwargs` (dict, optional): Additional arguments for cross-attention processing.

#### Return Value
- `StableDiffusionXLPipelineOutput`: An object containing the generated images.

--- 

This documentation provides a comprehensive overview of the `StableDiffusionXLOmostPipeline` class and its methods, detailing their purpose, parameters, and return values.

