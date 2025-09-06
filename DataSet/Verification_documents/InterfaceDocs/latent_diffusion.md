# API Documentation for DDIMSampler

## Class: DDIMSampler

### Description
The `DDIMSampler` class implements the Denoising Diffusion Implicit Models (DDIM) sampling algorithm. It is designed to generate samples from a trained diffusion model using a specified schedule.

### Attributes
- `model`: The diffusion model used for sampling.
- `ddpm_num_timesteps`: The number of timesteps in the DDPM (Denoising Diffusion Probabilistic Model).
- `schedule`: The schedule type for sampling (default is "linear").
- `ddim_timesteps`: The timesteps used for DDIM sampling.
- `betas`: The beta values for the diffusion process.
- `alphas_cumprod`: The cumulative product of alpha values.
- `alphas_cumprod_prev`: The cumulative product of previous alpha values.
- `sqrt_alphas_cumprod`: The square root of the cumulative product of alpha values.
- `sqrt_one_minus_alphas_cumprod`: The square root of one minus the cumulative product of alpha values.
- `log_one_minus_alphas_cumprod`: The logarithm of one minus the cumulative product of alpha values.
- `sqrt_recip_alphas_cumprod`: The square root of the reciprocal of the cumulative product of alpha values.
- `sqrt_recipm1_alphas_cumprod`: The square root of the reciprocal of the cumulative product of alpha values minus one.
- `ddim_sigmas`: The sigma values for DDIM sampling.
- `ddim_alphas`: The alpha values for DDIM sampling.
- `ddim_alphas_prev`: The previous alpha values for DDIM sampling.
- `ddim_sqrt_one_minus_alphas`: The square root of one minus the alpha values for DDIM sampling.
- `ddim_sigmas_for_original_num_steps`: The sigma values for the original number of sampling steps.

### Method: __init__

#### Description
Initializes the `DDIMSampler` with a given model and schedule.

#### Parameters
- `model`: The diffusion model to be used for sampling.
- `schedule` (str, optional): The schedule type for sampling (default is "linear").
- `**kwargs`: Additional keyword arguments.

#### Return Value
None

---

### Method: register_buffer

#### Description
Registers a tensor as a buffer in the module, ensuring it is moved to the correct device.

#### Parameters
- `name` (str): The name of the buffer.
- `attr` (torch.Tensor): The tensor to be registered as a buffer.

#### Return Value
None

---

### Method: make_schedule

#### Description
Creates a schedule for DDIM sampling based on the specified number of steps and other parameters.

#### Parameters
- `ddim_num_steps` (int): The number of DDIM sampling steps.
- `ddim_discretize` (str, optional): The discretization method for DDIM (default is "uniform").
- `ddim_eta` (float, optional): The eta parameter for DDIM (default is 0.0).
- `verbose` (bool, optional): If True, prints verbose output (default is True).

#### Return Value
None

---

### Method: sample

#### Description
Generates samples using the DDIM sampling method.

#### Parameters
- `S` (int): The number of sampling steps.
- `batch_size` (int): The number of samples to generate.
- `shape` (tuple): The shape of the generated samples (C, H, W).
- `conditioning` (optional): Conditioning information for the samples.
- `callback` (callable, optional): A callback function to be called at each step.
- `normals_sequence` (optional): A sequence of normal distributions.
- `img_callback` (callable, optional): A callback function for images.
- `quantize_x0` (bool, optional): If True, quantizes the denoised output (default is False).
- `eta` (float, optional): The eta parameter for DDIM (default is 0.0).
- `mask` (optional): A mask to apply to the samples.
- `x0` (optional): Initial input for the sampling process.
- `temperature` (float, optional): Temperature for sampling (default is 1.0).
- `noise_dropout` (float, optional): Dropout rate for noise (default is 0.0).
- `score_corrector` (optional): A score corrector for the sampling process.
- `corrector_kwargs` (optional): Additional arguments for the score corrector.
- `verbose` (bool, optional): If True, prints verbose output (default is True).
- `x_T` (optional): Initial noise for the sampling process.
- `log_every_t` (int, optional): Frequency of logging (default is 100).
- `unconditional_guidance_scale` (float, optional): Scale for unconditional guidance (default is 1.0).
- `unconditional_conditioning` (optional): Conditioning for unconditional guidance.
- `**kwargs`: Additional keyword arguments.

#### Return Value
- `samples`: The generated samples.
- `intermediates`: Intermediate results during sampling.

---

### Method: ddim_sampling

#### Description
Performs the actual DDIM sampling process.

#### Parameters
- `cond` (optional): Conditioning information for the samples.
- `shape` (tuple): The shape of the generated samples (C, H, W).
- `x_T` (optional): Initial noise for the sampling process.
- `ddim_use_original_steps` (bool, optional): If True, uses the original number of steps (default is False).
- `callback` (callable, optional): A callback function to be called at each step.
- `timesteps` (optional): Specific timesteps to use for sampling.
- `quantize_denoised` (bool, optional): If True, quantizes the denoised output (default is False).
- `mask` (optional): A mask to apply to the samples.
- `x0` (optional): Initial input for the sampling process.
- `img_callback` (callable, optional): A callback function for images.
- `log_every_t` (int, optional): Frequency of logging (default is 100).
- `temperature` (float, optional): Temperature for sampling (default is 1.0).
- `noise_dropout` (float, optional): Dropout rate for noise (default is 0.0).
- `score_corrector` (optional): A score corrector for the sampling process.
- `corrector_kwargs` (optional): Additional arguments for the score corrector.
- `unconditional_guidance_scale` (float, optional): Scale for unconditional guidance (default is 1.0).
- `unconditional_conditioning` (optional): Conditioning for unconditional guidance.

#### Return Value
- `img`: The generated image.
- `intermediates`: Intermediate results during sampling.

---

### Method: p_sample_ddim

#### Description
Performs a single step of the DDIM sampling process.

#### Parameters
- `x` (torch.Tensor): The current sample.
- `c` (optional): Conditioning information for the sample.
- `t` (int): The current timestep.
- `index` (int): The index of the current timestep.
- `repeat_noise` (bool, optional): If True, repeats noise for each sample (default is False).
- `use_original_steps` (bool, optional): If True, uses the original number of steps (default is False).
- `quantize_denoised` (bool, optional): If True, quantizes the denoised output (default is False).
- `temperature` (float, optional): Temperature for sampling (default is 1.0).
- `noise_dropout` (float, optional): Dropout rate for noise (default is 0.0).
- `score_corrector` (optional): A score corrector for the sampling process.
- `corrector_kwargs` (optional): Additional arguments for the score corrector.
- `unconditional_guidance_scale` (float, optional): Scale for unconditional guidance (default is 1.0).
- `unconditional_conditioning` (optional): Conditioning for unconditional guidance.

#### Return Value
- `x_prev`: The previous sample after applying the DDIM step.
- `pred_x0`: The predicted x0 value.

--- 

This documentation provides a comprehensive overview of the `DDIMSampler` class and its methods, detailing their parameters, return values, and purposes.

