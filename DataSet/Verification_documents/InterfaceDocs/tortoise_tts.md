# API Documentation

## Class: `TextToSpeech`
The `TextToSpeech` class implements a multi-stage text-to-speech (TTS) synthesis system based on deep learning, combining autoregressive models, diffusion models, and speech quality assessment models.

### Attributes:
- `models_dir`: Directory where model weights are stored.
- `autoregressive_batch_size`: Number of samples to generate per batch.
- `enable_redaction`: Flag to enable automatic redaction of text enclosed in brackets.
- `device`: Device to run the model (CPU or GPU).
- `tokenizer`: Tokenizer for processing input text.
- `half`: Flag to use half-precision for model inference.
- `autoregressive`: Autoregressive model for generating speech.
- `diffusion`: Diffusion model for refining generated speech.
- `clvp`: CLVP model for selecting the best output.
- `cvvp`: CVVP model for additional output selection (loaded lazily).
- `vocoder`: Vocoder for converting mel spectrograms to audio waveforms.
- `stft`: Short-time Fourier transform model (loaded lazily).
- `rlg_auto`: Random latent generator for autoregressive model (loaded lazily).
- `rlg_diffusion`: Random latent generator for diffusion model (loaded lazily).

### Method: `__init__(self, autoregressive_batch_size=None, models_dir=MODELS_DIR, enable_redaction=True, kv_cache=False, use_deepspeed=False, half=False, device=None, tokenizer_vocab_file=None, tokenizer_basic=False)`
#### Parameters:
- `autoregressive_batch_size` (int, optional): Specifies how many samples to generate per batch. Lower this if you encounter GPU OOM errors. Default is `None`.
- `models_dir` (str): Directory where model weights are stored. Default is `MODELS_DIR`.
- `enable_redaction` (bool): When true, text enclosed in brackets is automatically redacted from the spoken output. Default is `True`.
- `kv_cache` (bool): Flag to enable key-value caching. Default is `False`.
- `use_deepspeed` (bool): Flag to enable DeepSpeed optimization. Default is `False`.
- `half` (bool): Flag to use half-precision for model inference. Default is `False`.
- `device` (str, optional): Device to use when running the model. If omitted, the device will be automatically chosen.
- `tokenizer_vocab_file` (str, optional): Path to the tokenizer vocabulary file. Default is `None`.
- `tokenizer_basic` (bool): Flag to use basic text cleaners. Default is `False`.

#### Purpose:
Initializes the `TextToSpeech` instance, loading the necessary models and setting up the environment for TTS synthesis.

---

### Method: `temporary_cuda(self, model)`
#### Parameters:
- `model` (torch.nn.Module): The model to temporarily move to the CUDA device.

#### Returns:
- `torch.nn.Module`: The model moved to the specified device.

#### Purpose:
A context manager that temporarily moves a model to the CUDA device for inference and returns it to the CPU after use.

---

### Method: `load_cvvp(self)`
#### Parameters:
- None

#### Returns:
- None

#### Purpose:
Loads the CVVP model for additional output selection during TTS synthesis.

---

### Method: `get_conditioning_latents(self, voice_samples, return_mels=False)`
#### Parameters:
- `voice_samples` (list of torch.Tensor): List of 2 or more ~10 second reference clips, which should be torch tensors containing 22.05kHz waveform data.
- `return_mels` (bool, optional): If `True`, returns mel spectrograms along with conditioning latents. Default is `False`.

#### Returns:
- tuple: A tuple containing:
  - `autoregressive_conditioning_latent` (torch.Tensor): Latent representation for the autoregressive model.
  - `diffusion_conditioning_latent` (torch.Tensor): Latent representation for the diffusion model.
  - (optional) `auto_conds` (torch.Tensor): Conditioning tensors for autoregressive model.
  - (optional) `diffusion_conds` (torch.Tensor): Conditioning tensors for diffusion model.

#### Purpose:
Transforms one or more voice samples into expressive learned latents that encode aspects of the provided clips like voice, intonation, and acoustic properties.

---

### Method: `get_random_conditioning_latents(self)`
#### Parameters:
- None

#### Returns:
- tuple: A tuple containing:
  - `autoregressive_conditioning_latent` (torch.Tensor): Random latent for the autoregressive model.
  - `diffusion_conditioning_latent` (torch.Tensor): Random latent for the diffusion model.

#### Purpose:
Generates random conditioning latents for the autoregressive and diffusion models.

---

### Method: `tts_with_preset(self, text, preset='fast', **kwargs)`
#### Parameters:
- `text` (str): Text to be spoken.
- `preset` (str, optional): Preset generation parameters. Options are 'ultra_fast', 'fast', 'standard', 'high_quality'. Default is 'fast'.
- `**kwargs`: Additional keyword arguments to override preset settings.

#### Returns:
- torch.Tensor: Generated audio clip(s) as a torch tensor.

#### Purpose:
Calls TTS with one of a set of preset generation parameters for convenience.

---

### Method: `tts(self, text, voice_samples=None, conditioning_latents=None, k=1, verbose=True, use_deterministic_seed=None, return_deterministic_state=False, num_autoregressive_samples=512, temperature=.8, length_penalty=1, repetition_penalty=2.0, top_p=.8, max_mel_tokens=500, cvvp_amount=.0, diffusion_iterations=100, cond_free=True, cond_free_k=2, diffusion_temperature=1.0, **hf_generate_kwargs)`
#### Parameters:
- `text` (str): Text to be spoken.
- `voice_samples` (list of torch.Tensor, optional): List of 2 or more ~10 second reference clips.
- `conditioning_latents` (tuple, optional): A tuple of (autoregressive_conditioning_latent, diffusion_conditioning_latent).
- `k` (int, optional): Number of returned clips. Default is `1`.
- `verbose` (bool, optional): Whether to print log messages. Default is `True`.
- `use_deterministic_seed` (int, optional): Seed for reproducibility. Default is `None`.
- `return_deterministic_state` (bool, optional): If `True`, returns the deterministic state. Default is `False`.
- `num_autoregressive_samples` (int, optional): Number of samples from the autoregressive model. Default is `512`.
- `temperature` (float, optional): Softmax temperature of the autoregressive model. Default is `0.8`.
- `length_penalty` (float, optional): Length penalty applied to the autoregressive decoder. Default is `1`.
- `repetition_penalty` (float, optional): Penalty to prevent repetition. Default is `2.0`.
- `top_p` (float, optional): P value for nucleus sampling. Default is `0.8`.
- `max_mel_tokens` (int, optional): Restricts output length. Default is `500`.
- `cvvp_amount` (float, optional): Influence of the CVVP model. Default is `0.0`.
- `diffusion_iterations` (int, optional): Number of diffusion steps. Default is `100`.
- `cond_free` (bool, optional): Whether to perform conditioning-free diffusion. Default is `True`.
- `cond_free_k` (float, optional): Balances conditioning-free signal with conditioning-present signal. Default is `2`.
- `diffusion_temperature` (float, optional): Controls variance of noise fed into the diffusion model. Default is `1.0`.
- `**hf_generate_kwargs`: Extra keyword arguments for the Hugging Face Transformers generate API.

#### Returns:
- torch.Tensor: Generated audio clip(s) as a torch tensor.

#### Purpose:
Produces an audio clip of the given text being spoken with the specified reference voice.

---

### Method: `deterministic_state(self, seed=None)`
#### Parameters:
- `seed` (int, optional): Seed for random number generation. Default is `None`.

#### Returns:
- int: The seed used for random number generation.

#### Purpose:
Sets the random seeds for reproducibility and returns the seed used.