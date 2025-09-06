# API Documentation

## Functions

### `load_module_gpu`

**Description**: Loads a model onto the GPU for processing.

**Parameters**:
- `model` (torch.nn.Module): The model to be loaded onto the GPU.

**Return Value**: None

**Purpose**: This function transfers the specified model to the GPU, enabling faster computations.

---

### `unload_module_gpu`

**Description**: Unloads a model from the GPU and clears the CUDA cache.

**Parameters**:
- `model` (torch.nn.Module): The model to be unloaded from the GPU.

**Return Value**: None

**Purpose**: This function transfers the specified model back to the CPU and clears the GPU memory cache to free up resources.

---

### `initial_model_load`

**Description**: Initializes the model by converting it to half-precision.

**Parameters**:
- `model` (torch.nn.Module): The model to be initialized.

**Return Value**: torch.nn.Module: The initialized model in half-precision.

**Purpose**: This function prepares the model for inference by converting its parameters to half-precision, which can improve performance on compatible hardware.

---

### `preprocess_video`

**Description**: Preprocesses a video or a set of images for further processing.

**Parameters**:
- `input_path` (str): Path to the input video file or directory containing images.
- `remove_bg` (bool, optional): Whether to remove the background from images (default is False).
- `n_frames` (int, optional): Number of frames to process (default is 21).
- `W` (int, optional): Width of the output frames (default is 576).
- `H` (int, optional): Height of the output frames (default is 576).
- `output_folder` (str, optional): Directory to save the processed video (default is the same as input path).
- `image_frame_ratio` (float, optional): Aspect ratio for the output frames (default is 0.917).
- `base_count` (int, optional): Base count for naming the output file (default is 0).

**Return Value**: str: Path to the processed video file.

**Purpose**: This function reads a video or a set of images, optionally removes backgrounds, crops the frames, and saves the processed frames as a video file.

---

### `do_sample`

**Description**: Samples from the model using a specified sampler.

**Parameters**:
- `model` (torch.nn.Module): The model to sample from.
- `sampler` (callable): The sampling method to use.
- `value_dict` (dict): Dictionary containing input values for the model.
- `num_samples` (int): Number of samples to generate.
- `H` (int): Height of the output samples.
- `W` (int): Width of the output samples.
- `C` (int): Number of channels in the output samples.
- `F` (int): Factor for downsampling.
- `force_uc_zero_embeddings` (list, optional): List of embeddings to force to zero (default is None).
- `force_cond_zero_embeddings` (list, optional): List of conditional embeddings to force to zero (default is None).
- `batch2model_input` (list, optional): Additional inputs for the model (default is None).
- `return_latents` (bool, optional): Whether to return latent representations (default is False).
- `filter` (callable, optional): Optional filtering function (default is None).
- `T` (int, optional): Number of timesteps (default is None).
- `additional_batch_uc_fields` (list, optional): Additional fields for unconditional sampling (default is None).
- `decoding_t` (int, optional): Decoding timesteps (default is None).

**Return Value**: torch.Tensor: The generated samples.

**Purpose**: This function performs sampling from the model using the specified sampler and returns the generated samples.

---

### `run_img2vid`

**Description**: Generates a video from a single image using the model.

**Parameters**:
- `version_dict` (dict): Dictionary containing model version and configuration options.
- `model` (torch.nn.Module): The model to use for generating the video.
- `image` (torch.Tensor): Input image tensor.
- `seed` (int, optional): Random seed for reproducibility (default is 23).
- `polar_rad` (list, optional): List of polar angles for motion (default is [10] * 21).
- `azim_rad` (numpy.ndarray, optional): Array of azimuth angles for motion (default is np.linspace(0, 360, 21 + 1)[1:]).
- `cond_motion` (torch.Tensor, optional): Conditional motion tensor (default is None).
- `cond_view` (torch.Tensor, optional): Conditional view tensor (default is None).
- `decoding_t` (int, optional): Decoding timesteps (default is None).
- `cond_mv` (bool, optional): Whether to condition on motion (default is True).

**Return Value**: torch.Tensor: The generated video samples.

**Purpose**: This function generates a video from the provided image by utilizing the model and specified parameters for motion and view conditions.

---

### `load_model`

**Description**: Loads a model from a configuration file and initializes it.

**Parameters**:
- `config` (str): Path to the model configuration file.
- `device` (str): Device to load the model on (e.g., "cuda" or "cpu").
- `num_frames` (int): Number of frames for the model to process.
- `num_steps` (int): Number of steps for the sampler.
- `verbose` (bool, optional): Whether to print verbose output (default is False).
- `ckpt_path` (str, optional): Path to the model checkpoint (default is None).

**Return Value**: Tuple[torch.nn.Module, DeepFloydDataFiltering]: The loaded model and the data filtering instance.

**Purpose**: This function loads a model based on the provided configuration and initializes it for inference, returning both the model and a data filtering instance.