# API Documentation for AnimationPipeline

## Class: AnimationPipeline

The `AnimationPipeline` class is designed to facilitate the generation of animated videos using a diffusion model. It integrates various components such as a variational autoencoder (VAE), a text encoder, a tokenizer, a UNet model, and a scheduler to produce high-quality animations based on textual prompts.

### Attributes:
- **vae**: An instance of `AutoencoderKL`, responsible for encoding and decoding video latents.
- **text_encoder**: An instance of `CLIPTextModel`, used for encoding text prompts into embeddings.
- **tokenizer**: An instance of `CLIPTokenizer`, used for tokenizing input text.
- **unet**: An instance of `UNet3DConditionModel`, which performs the core denoising process.
- **scheduler**: An instance of a scheduler (e.g., `DDIMScheduler`, `PNDMScheduler`, etc.) that manages the diffusion process.
- **controlnet**: An optional instance of `SparseControlNetModel`, used for additional control over the generation process.
- **vae_scale_factor**: A scaling factor derived from the VAE configuration, used to adjust the dimensions of the generated video.

### Method: `__init__`

```python
def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, unet: UNet3DConditionModel, scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler], controlnet: Union[SparseControlNetModel, None] = None):
```

#### Parameters:
- **vae**: An instance of `AutoencoderKL` for encoding and decoding video latents.
- **text_encoder**: An instance of `CLIPTextModel` for encoding text prompts.
- **tokenizer**: An instance of `CLIPTokenizer` for tokenizing input text.
- **unet**: An instance of `UNet3DConditionModel` for the denoising process.
- **scheduler**: An instance of a scheduler for managing the diffusion process.
- **controlnet**: (Optional) An instance of `SparseControlNetModel` for additional control over the generation process.

#### Return Value:
- Initializes an instance of the `AnimationPipeline` class.

#### Purpose:
This method initializes the `AnimationPipeline` with the necessary components for generating animated videos.

---

### Method: `enable_vae_slicing`

```python
def enable_vae_slicing(self):
```

#### Parameters:
- None

#### Return Value:
- None

#### Purpose:
Enables VAE slicing, which allows for more efficient memory usage during the decoding process.

---

### Method: `disable_vae_slicing`

```python
def disable_vae_slicing(self):
```

#### Parameters:
- None

#### Return Value:
- None

#### Purpose:
Disables VAE slicing, reverting to the default decoding behavior.

---

### Method: `enable_sequential_cpu_offload`

```python
def enable_sequential_cpu_offload(self, gpu_id=0):
```

#### Parameters:
- **gpu_id**: (Optional) An integer specifying the GPU ID to use for offloading. Default is `0`.

#### Return Value:
- None

#### Purpose:
Enables sequential CPU offloading for the models in the pipeline, allowing for more efficient memory management on the GPU.

---

### Method: `decode_latents`

```python
def decode_latents(self, latents):
```

#### Parameters:
- **latents**: A tensor containing the latent representations to be decoded.

#### Return Value:
- Returns a numpy array representing the decoded video.

#### Purpose:
Decodes the latent representations into video frames using the VAE.

---

### Method: `prepare_extra_step_kwargs`

```python
def prepare_extra_step_kwargs(self, generator, eta):
```

#### Parameters:
- **generator**: A random number generator or a list of generators for reproducibility.
- **eta**: A float value used for controlling the randomness in the DDIM scheduler.

#### Return Value:
- Returns a dictionary containing extra keyword arguments for the scheduler step.

#### Purpose:
Prepares additional parameters required for the scheduler's step function, accommodating different scheduler signatures.

---

### Method: `check_inputs`

```python
def check_inputs(self, prompt, height, width, callback_steps):
```

#### Parameters:
- **prompt**: A string or list of strings representing the input prompt(s).
- **height**: An integer specifying the height of the generated video.
- **width**: An integer specifying the width of the generated video.
- **callback_steps**: An integer specifying the frequency of callback execution.

#### Return Value:
- Raises a `ValueError` if any input validation fails.

#### Purpose:
Validates the input parameters to ensure they meet the required criteria for video generation.

---

### Method: `prepare_latents`

```python
def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
```

#### Parameters:
- **batch_size**: An integer specifying the number of videos to generate.
- **num_channels_latents**: An integer specifying the number of channels in the latent representation.
- **video_length**: An integer specifying the length of the video in frames.
- **height**: An integer specifying the height of the video.
- **width**: An integer specifying the width of the video.
- **dtype**: The data type of the latents (e.g., `torch.float32`).
- **device**: The device on which to allocate the latents (e.g., CPU or GPU).
- **generator**: A random number generator or a list of generators for reproducibility.
- **latents**: (Optional) A tensor of existing latents to be used instead of generating new ones.

#### Return Value:
- Returns a tensor containing the prepared latents.

#### Purpose:
Prepares the latent representations for the video generation process, either by generating new latents or using provided ones.

---

### Method: `__call__`

```python
def __call__(self, prompt: Union[str, List[str]], video_length: Optional[int], height: Optional[int] = None, width: Optional[int] = None, num_inference_steps: int = 50, guidance_scale: float = 7.5, negative_prompt: Optional[Union[str, List[str]]] = None, num_videos_per_prompt: Optional[int] = 1, eta: float = 0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, latents: Optional[torch.FloatTensor] = None, output_type: Optional[str] = "tensor", return_dict: bool = True, callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None, callback_steps: Optional[int] = 1, controlnet_images: torch.FloatTensor = None, controlnet_image_index: list = [0], controlnet_conditioning_scale: Union[float, List[float]] = 1.0, **kwargs):
```

#### Parameters:
- **prompt**: A string or list of strings representing the input prompt(s).
- **video_length**: An integer specifying the length of the video in frames.
- **height**: (Optional) An integer specifying the height of the generated video.
- **width**: (Optional) An integer specifying the width of the generated video.
- **num_inference_steps**: An integer specifying the number of inference steps for the diffusion process. Default is `50`.
- **guidance_scale**: A float specifying the guidance scale for classifier-free guidance. Default is `7.5`.
- **negative_prompt**: (Optional) A string or list of strings for negative prompts.
- **num_videos_per_prompt**: (Optional) An integer specifying the number of videos to generate per prompt. Default is `1`.
- **eta**: A float value used for controlling randomness in the DDIM scheduler. Default is `0.0`.
- **generator**: (Optional) A random number generator or a list of generators for reproducibility.
- **latents**: (Optional) A tensor of existing latents to be used instead of generating new ones.
- **output_type**: (Optional) A string specifying the output type, either "tensor" or "numpy". Default is "tensor".
- **return_dict**: (Optional) A boolean indicating whether to return a dictionary or just the video. Default is `True`.
- **callback**: (Optional) A callable function for progress callbacks.
- **callback_steps**: (Optional) An integer specifying the frequency of callback execution.
- **controlnet_images**: (Optional) A tensor of images for controlnet conditioning.
- **controlnet_image_index**: (Optional) A list of indices for controlnet images. Default is `[0]`.
- **controlnet_conditioning_scale**: (Optional) A float or list of floats for controlnet conditioning scale. Default is `1.0`.

#### Return Value:
- Returns an instance of `AnimationPipelineOutput` containing the generated videos, or just the video if `return_dict` is `False`.

#### Purpose:
Generates animated videos based on the provided prompts and parameters, utilizing the configured components of the pipeline.

