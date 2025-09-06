# API Documentation for MusicGen

## Class: MusicGen

The `MusicGen` class provides a convenient API for generating music based on text descriptions and optional melody conditioning. It allows users to set various parameters for music generation, including duration and sampling options.

### Attributes
- **name (str)**: The name of the model.
- **compression_model (CompressionModel)**: The compression model used to map audio to invertible discrete representations.
- **lm (LMModel)**: The language model over discrete representations.
- **max_duration (float, optional)**: The maximum duration the model can produce, otherwise inferred from the training parameters.
- **duration (float)**: The duration of the generated waveform, default is set to 15 seconds.
- **generation_params (dict)**: A dictionary containing parameters for music generation.
- **extend_stride (float)**: The stride used for extended generation beyond the maximum duration.

### Method: __init__

```python
def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel, max_duration: tp.Optional[float] = None):
```

#### Parameters
- **name (str)**: The name of the model.
- **compression_model (CompressionModel)**: The compression model used for audio representation.
- **lm (LMModel)**: The language model for generating music.
- **max_duration (float, optional)**: Maximum duration for generated music. If not provided, inferred from training parameters.

#### Return Value
- Initializes a `MusicGen` instance.

#### Purpose
- Constructs a `MusicGen` object with specified model parameters and sets a default generation duration.

---

### Method: get_pretrained

```python
@staticmethod
def get_pretrained(name: str = 'facebook/musicgen-melody', device=None):
```

#### Parameters
- **name (str, optional)**: The name of the pretrained model to load. Options include:
  - 'facebook/musicgen-small'
  - 'facebook/musicgen-medium'
  - 'facebook/musicgen-melody'
  - 'facebook/musicgen-large'
  - 'facebook/musicgen-style'
- **device (str, optional)**: The device to load the model on ('cuda' or 'cpu'). If not specified, it defaults to 'cuda' if available.

#### Return Value
- Returns an instance of `MusicGen` initialized with the specified pretrained model.

#### Purpose
- Loads a pretrained `MusicGen` model based on the specified name and device.

---

### Method: set_generation_params

```python
def set_generation_params(self, use_sampling: bool = True, top_k: int = 250, top_p: float = 0.0, temperature: float = 1.0, duration: float = 30.0, cfg_coef: float = 3.0, cfg_coef_beta: tp.Optional[float] = None, two_step_cfg: bool = False, extend_stride: float = 18):
```

#### Parameters
- **use_sampling (bool, optional)**: If True, uses sampling; otherwise, uses argmax decoding. Defaults to True.
- **top_k (int, optional)**: The number of top tokens to sample from. Defaults to 250.
- **top_p (float, optional)**: The cumulative probability threshold for sampling. Defaults to 0.0.
- **temperature (float, optional)**: Softmax temperature parameter. Defaults to 1.0.
- **duration (float, optional)**: Duration of the generated waveform. Defaults to 30.0 seconds.
- **cfg_coef (float, optional)**: Coefficient for classifier-free guidance. Defaults to 3.0.
- **cfg_coef_beta (float, optional)**: Beta coefficient for double classifier-free guidance. Optional.
- **two_step_cfg (bool, optional)**: If True, performs two forward passes for classifier-free guidance. Defaults to False.
- **extend_stride (float)**: The stride for extended generation. Must be less than `max_duration`.

#### Return Value
- None

#### Purpose
- Sets the parameters for music generation, allowing customization of the generation process.

---

### Method: set_style_conditioner_params

```python
def set_style_conditioner_params(self, eval_q: int = 3, excerpt_length: float = 3.0, ds_factor: tp.Optional[int] = None, encodec_n_q: tp.Optional[int] = None) -> None:
```

#### Parameters
- **eval_q (int)**: The number of residual quantization streams for style conditioning. Smaller values create a narrower information bottleneck. Defaults to 3.
- **excerpt_length (float)**: The length of the audio excerpt used for conditioning in seconds. Defaults to 3.0.
- **ds_factor (int, optional)**: The downsampling factor for style tokens. Optional.
- **encodec_n_q (int, optional)**: Number of streams for feature extraction if encodec is used. Optional.

#### Return Value
- None

#### Purpose
- Sets parameters for the style conditioner, applicable only for the MusicGen-Style model.

---

### Method: generate_with_chroma

```python
def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType, melody_sample_rate: int, progress: bool = False, return_tokens: bool = False) -> tp.Union[torch.Tensor, tp.Tuple[torch.Tensor, torch.Tensor]]:
```

#### Parameters
- **descriptions (list of str)**: A list of text descriptions used for conditioning the music generation.
- **melody_wavs (MelodyType)**: A batch of melody waveforms used for conditioning. Can be a tensor of shape [B, C, T] or a list of tensors.
- **melody_sample_rate (int)**: The sample rate of the melody waveforms.
- **progress (bool, optional)**: If True, displays progress during generation. Defaults to False.
- **return_tokens (bool, optional)**: If True, returns the generated tokens along with the audio. Defaults to False.

#### Return Value
- Returns a tensor of generated audio of shape [B, C, T]. If `return_tokens` is True, returns a tuple of (audio tensor, tokens tensor).

#### Purpose
- Generates music samples conditioned on text descriptions and optional melody inputs.