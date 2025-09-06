# API Documentation

## Function: `karras_sample`

### Description
The `karras_sample` function generates samples from a diffusion model using the Karras sampling method. It allows for various sampling strategies and provides options for noise control and denoising.

### Parameters

- **diffusion** (`DiffusionModel`): The diffusion model used for denoising the samples.
  
- **model** (`nn.Module`): The neural network model that will be used for generating the samples.

- **shape** (`tuple`): The shape of the generated samples, typically in the format `(batch_size, channels, height, width)`.

- **steps** (`int`): The number of sampling steps to perform. Must be a positive integer.

- **clip_denoised** (`bool`, optional): If `True`, the denoised samples will be clamped to the range [-1, 1]. Default is `True`.

- **progress** (`bool`, optional): If `True`, progress will be displayed during sampling. Default is `False`.

- **callback** (`callable`, optional): A callback function that will be called at each sampling step. Default is `None`.

- **model_kwargs** (`dict`, optional): Additional keyword arguments to pass to the model during denoising. Default is `None`.

- **device** (`torch.device`, optional): The device on which to perform computations (e.g., CPU or GPU). Default is `None`, which uses the default device.

- **sigma_min** (`float`, optional): The minimum noise level for the sampling process. Default is `0.002`. Must be a positive float.

- **sigma_max** (`float`, optional): The maximum noise level for the sampling process. Default is `80.0`. Must be a positive float.

- **rho** (`float`, optional): A parameter that controls the distribution of noise levels. Default is `7.0`. Must be a positive float.

- **sampler** (`str`, optional): The sampling method to use. Options include "heun", "dpm", "ancestral", "onestep", "progdist", "euler", and "multistep". Default is "heun".

- **s_churn** (`float`, optional): Controls the amount of noise added during sampling. Default is `0.0`. Must be a non-negative float.

- **s_tmin** (`float`, optional): The minimum time step for sampling. Default is `0.0`. Must be a non-negative float.

- **s_tmax** (`float`, optional): The maximum time step for sampling. Default is `float("inf")`. Must be a non-negative float.

- **s_noise** (`float`, optional): The amount of noise to add during sampling. Default is `1.0`. Must be a non-negative float.

- **generator** (`torch.Generator`, optional): A random number generator for reproducibility. Default is `None`, which uses a dummy generator.

- **ts** (`torch.Tensor`, optional): A tensor of time steps for multistep sampling. Default is `None`.

### Returns
- **torch.Tensor**: A tensor containing the generated samples, clamped to the range [-1, 1].

### Purpose
The `karras_sample` function is designed to generate high-quality samples from a diffusion model using various sampling strategies. It provides flexibility in terms of noise control and allows for the integration of custom models and callbacks during the sampling process.

