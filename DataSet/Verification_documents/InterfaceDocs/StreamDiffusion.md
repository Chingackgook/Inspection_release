# API Documentation

## Class: StreamDiffusionWrapper

The `StreamDiffusionWrapper` class is designed to facilitate the use of the StreamDiffusion model for image generation tasks, including both text-to-image (txt2img) and image-to-image (img2img) generation. It provides methods to prepare the model for inference and to generate images based on provided prompts or input images.

### Attributes
- **sd_turbo** (bool): Indicates whether the model is a turbo version based on the model ID or path.
- **device** (Literal["cpu", "cuda"]): The device to use for inference (default is "cuda").
- **dtype** (torch.dtype): The data type for inference (default is `torch.float16`).
- **width** (int): The width of the generated image (default is 512).
- **height** (int): The height of the generated image (default is 512).
- **mode** (Literal["img2img", "txt2img"]): The mode of operation, either "img2img" or "txt2img" (default is "img2img").
- **output_type** (Literal["pil", "pt", "np", "latent"]): The output type of the generated image (default is "pil").
- **frame_buffer_size** (int): The size of the frame buffer for denoising batch (default is 1).
- **batch_size** (int): The batch size for inference, calculated based on `t_index_list` and `frame_buffer_size`.
- **use_denoising_batch** (bool): Indicates whether to use denoising batch (default is True).
- **use_safety_checker** (bool): Indicates whether to use a safety checker (default is False).
- **stream** (StreamDiffusion): An instance of the StreamDiffusion model loaded with the specified parameters.

### Method: `__init__`

```python
def __init__(
    self,
    model_id_or_path: str,
    t_index_list: List[int],
    lora_dict: Optional[Dict[str, float]] = None,
    mode: Literal["img2img", "txt2img"] = "img2img",
    output_type: Literal["pil", "pt", "np", "latent"] = "pil",
    lcm_lora_id: Optional[str] = None,
    vae_id: Optional[str] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    dtype: torch.dtype = torch.float16,
    frame_buffer_size: int = 1,
    width: int = 512,
    height: int = 512,
    warmup: int = 10,
    acceleration: Literal["none", "xformers", "tensorrt"] = "tensorrt",
    do_add_noise: bool = True,
    device_ids: Optional[List[int]] = None,
    use_lcm_lora: bool = True,
    use_tiny_vae: bool = True,
    enable_similar_image_filter: bool = False,
    similar_image_filter_threshold: float = 0.98,
    similar_image_filter_max_skip_frame: int = 10,
    use_denoising_batch: bool = True,
    cfg_type: Literal["none", "full", "self", "initialize"] = "self",
    seed: int = 2,
    use_safety_checker: bool = False,
    engine_dir: Optional[Union[str, Path]] = "engines",
)
```

#### Parameters
- **model_id_or_path** (str): The model ID or path to load.
- **t_index_list** (List[int]): A list of indices to use for inference.
- **lora_dict** (Optional[Dict[str, float]]): A dictionary mapping LoRA names to their scales (default is None).
- **mode** (Literal["img2img", "txt2img"]): The mode of operation (default is "img2img").
- **output_type** (Literal["pil", "pt", "np", "latent"]): The output type of the generated image (default is "pil").
- **lcm_lora_id** (Optional[str]): The ID for LCM-LoRA (default is None).
- **vae_id** (Optional[str]): The ID for the VAE (default is None).
- **device** (Literal["cpu", "cuda"]): The device to use for inference (default is "cuda").
- **dtype** (torch.dtype): The data type for inference (default is `torch.float16`).
- **frame_buffer_size** (int): The size of the frame buffer for denoising batch (default is 1).
- **width** (int): The width of the generated image (default is 512).
- **height** (int): The height of the generated image (default is 512).
- **warmup** (int): The number of warmup steps to perform (default is 10).
- **acceleration** (Literal["none", "xformers", "tensorrt"]): The acceleration method (default is "tensorrt").
- **do_add_noise** (bool): Whether to add noise for denoising steps (default is True).
- **device_ids** (Optional[List[int]]): The device IDs for DataParallel (default is None).
- **use_lcm_lora** (bool): Whether to use LCM-LoRA (default is True).
- **use_tiny_vae** (bool): Whether to use TinyVAE (default is True).
- **enable_similar_image_filter** (bool): Whether to enable similar image filtering (default is False).
- **similar_image_filter_threshold** (float): The threshold for similar image filtering (default is 0.98).
- **similar_image_filter_max_skip_frame** (int): The maximum number of frames to skip for similar image filtering (default is 10).
- **use_denoising_batch** (bool): Whether to use denoising batch (default is True).
- **cfg_type** (Literal["none", "full", "self", "initialize"]): The configuration type for img2img mode (default is "self").
- **seed** (int): The seed for random number generation (default is 2).
- **use_safety_checker** (bool): Whether to use a safety checker (default is False).
- **engine_dir** (Optional[Union[str, Path]]): The directory for engine files (default is "engines").

#### Returns
- None

#### Purpose
Initializes the `StreamDiffusionWrapper` instance with the specified parameters, loading the necessary models and configurations for image generation.

---

### Method: `prepare`

```python
def prepare(
    self,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 1.2,
    delta: float = 1.0,
) -> None
```

#### Parameters
- **prompt** (str): The prompt to generate images from.
- **negative_prompt** (str): The negative prompt to guide the generation (default is an empty string).
- **num_inference_steps** (int): The number of inference steps to perform (default is 50).
- **guidance_scale** (float): The guidance scale to use (default is 1.2).
- **delta** (float): The delta multiplier for virtual residual noise (default is 1.0).

#### Returns
- None

#### Purpose
Prepares the model for inference by setting the prompt, negative prompt, and other parameters related to the image generation process.

---

### Method: `__call__`

```python
def __call__(
    self,
    image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    prompt: Optional[str] = None,
) -> Union[Image.Image, List[Image.Image]]
```

#### Parameters
- **image** (Optional[Union[str, Image.Image, torch.Tensor]]): The image to generate from (default is None).
- **prompt** (Optional[str]): The prompt to generate images from (default is None).

#### Returns
- **Union[Image.Image, List[Image.Image]]**: The generated image or a list of generated images.

#### Purpose
Performs image generation based on the specified mode (img2img or txt2img). If in img2img mode, it generates images based on the provided input image; if in txt2img mode, it generates images based on the provided prompt.

