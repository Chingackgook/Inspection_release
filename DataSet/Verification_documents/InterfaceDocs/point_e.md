# API Documentation for PointCloudSampler

## Class: PointCloudSampler

### Description
A wrapper around a model or stack of models that produces conditional or unconditional sample tensors. This class facilitates sampling from a single- or multi-stage point cloud diffusion model.

### Attributes
- **device** (`torch.device`): The device on which the computations will be performed (e.g., CPU or GPU).
- **num_points** (`Sequence[int]`): A sequence of integers representing the number of points to sample at each stage.
- **aux_channels** (`Sequence[str]`): A sequence of auxiliary channel names.
- **model_kwargs_key_filter** (`Sequence[str]`): A sequence of keys to filter model keyword arguments.
- **guidance_scale** (`Sequence[float]`): A sequence of floats representing the guidance scale for each model.
- **clip_denoised** (`bool`): A boolean indicating whether to clip the denoised output.
- **use_karras** (`Sequence[bool]`): A sequence of booleans indicating whether to use Karras sampling for each model.
- **karras_steps** (`Sequence[int]`): A sequence of integers representing the number of Karras steps for each model.
- **sigma_min** (`Sequence[float]`): A sequence of floats representing the minimum sigma values for each model.
- **sigma_max** (`Sequence[float]`): A sequence of floats representing the maximum sigma values for each model.
- **s_churn** (`Sequence[float]`): A sequence of floats representing the churn parameter for each model.
- **models** (`Sequence[nn.Module]`): A sequence of model instances used for sampling.
- **diffusions** (`Sequence[GaussianDiffusion]`): A sequence of diffusion instances corresponding to the models.

### Method: __init__

#### Description
Initializes a PointCloudSampler instance with the specified parameters.

#### Parameters
- **device** (`torch.device`): The device for computation.
- **models** (`Sequence[nn.Module]`): A sequence of model instances.
- **diffusions** (`Sequence[GaussianDiffusion]`): A sequence of diffusion instances.
- **num_points** (`Sequence[int]`): A sequence of integers for the number of points to sample.
- **aux_channels** (`Sequence[str]`): A sequence of auxiliary channel names.
- **model_kwargs_key_filter** (`Sequence[str]`, optional): A sequence of keys to filter model keyword arguments. Default is `("*",)`.
- **guidance_scale** (`Sequence[float]`, optional): A sequence of floats for guidance scale. Default is `(3.0, 3.0)`.
- **clip_denoised** (`bool`, optional): Whether to clip the denoised output. Default is `True`.
- **use_karras** (`Sequence[bool]`, optional): A sequence of booleans for Karras sampling. Default is `(True, True)`.
- **karras_steps** (`Sequence[int]`, optional): A sequence of integers for Karras steps. Default is `(64, 64)`.
- **sigma_min** (`Sequence[float]`, optional): A sequence of floats for minimum sigma values. Default is `(1e-3, 1e-3)`.
- **sigma_max** (`Sequence[float]`, optional): A sequence of floats for maximum sigma values. Default is `(120, 160)`.
- **s_churn** (`Sequence[float]`, optional): A sequence of floats for the churn parameter. Default is `(3, 0)`.

#### Return Value
None

---

### Method: num_stages

#### Description
Returns the number of stages in the sampler, which corresponds to the number of models.

#### Parameters
None

#### Return Value
- **int**: The number of stages (models) in the sampler.

---

### Method: sample_batch

#### Description
Samples a batch of point clouds using the specified model keyword arguments.

#### Parameters
- **batch_size** (`int`): The number of samples to generate.
- **model_kwargs** (`Dict[str, Any]`): A dictionary of keyword arguments to pass to the model.

#### Return Value
- **torch.Tensor**: A tensor containing the sampled point clouds.

---

### Method: sample_batch_progressive

#### Description
Samples a batch of point clouds progressively through each stage of the model.

#### Parameters
- **batch_size** (`int`): The number of samples to generate.
- **model_kwargs** (`Dict[str, Any]`): A dictionary of keyword arguments to pass to the model.

#### Return Value
- **Iterator[torch.Tensor]**: An iterator yielding tensors of sampled point clouds at each stage.

---

### Method: combine

#### Description
Combines multiple PointCloudSampler instances into a single sampler.

#### Parameters
- **samplers** (`*PointCloudSampler`): A variable number of PointCloudSampler instances to combine.

#### Return Value
- **PointCloudSampler**: A new PointCloudSampler instance that combines the provided samplers.

---

### Method: split_model_output

#### Description
Splits the model output into position and auxiliary channels.

#### Parameters
- **output** (`torch.Tensor`): The output tensor from the model.
- **rescale_colors** (`bool`, optional): Whether to rescale color values. Default is `False`.

#### Return Value
- **Tuple[torch.Tensor, Dict[str, torch.Tensor]]**: A tuple containing the position tensor and a dictionary of auxiliary channel tensors.

---

### Method: output_to_point_clouds

#### Description
Converts the model output tensor into a list of PointCloud instances.

#### Parameters
- **output** (`torch.Tensor`): The output tensor from the model.

#### Return Value
- **List[PointCloud]**: A list of PointCloud instances created from the output tensor.

---

### Method: with_options

#### Description
Creates a new PointCloudSampler instance with modified options.

#### Parameters
- **guidance_scale** (`float`): The guidance scale to use.
- **clip_denoised** (`bool`): Whether to clip the denoised output.
- **use_karras** (`Sequence[bool]`, optional): A sequence of booleans for Karras sampling. Default is `(True, True)`.
- **karras_steps** (`Sequence[int]`, optional): A sequence of integers for Karras steps. Default is `(64, 64)`.
- **sigma_min** (`Sequence[float]`, optional): A sequence of floats for minimum sigma values. Default is `(1e-3, 1e-3)`.
- **sigma_max** (`Sequence[float]`, optional): A sequence of floats for maximum sigma values. Default is `(120, 160)`.
- **s_churn** (`Sequence[float]`, optional): A sequence of floats for the churn parameter. Default is `(3, 0)`.

#### Return Value
- **PointCloudSampler**: A new PointCloudSampler instance with the specified options.

