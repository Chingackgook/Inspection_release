```markdown
# API Documentation for PULSE

## Class: PULSE

The `PULSE` class implements the PULSE (Photo Upsampling via Latent Space Exploration) model, which generates high-resolution images from low-resolution references by optimizing latent variables and noise tensors.

### Attributes:
- **synthesis**: An instance of the `G_synthesis` class, responsible for generating images from latent vectors.
- **verbose**: A boolean flag indicating whether to print detailed logs during execution.
- **gaussian_fit**: A dictionary containing the mean and standard deviation of the latent space distribution, used for normalizing the latent vectors.

### Method: `__init__(self, cache_dir, verbose=True)`

#### Parameters:
- **cache_dir** (str): The directory where model weights and other cached files will be stored.
- **verbose** (bool, optional): If set to `True`, detailed logs will be printed during the loading of the model. Default is `True`.

#### Return Value:
- None

#### Purpose:
Initializes the PULSE model by loading the synthesis and mapping networks, and computes the Gaussian fit for the latent space if it is not already cached.

---

### Method: `forward(self, ref_im, seed, loss_str, eps, noise_type, num_trainable_noise_layers, tile_latent, bad_noise_layers, opt_name, learning_rate, steps, lr_schedule, save_intermediate, **kwargs)`

#### Parameters:
- **ref_im** (torch.Tensor): A tensor containing the reference images to guide the generation process. Shape: (batch_size, channels, height, width).
- **seed** (int or None): A seed for random number generation. If `None`, randomness is not fixed.
- **loss_str** (str): A string specifying the type of loss function to use during optimization.
- **eps** (float): The threshold for the L2 loss; optimization will stop if the L2 loss is below this value.
- **noise_type** (str): Specifies the type of noise to use. Options are 'zero', 'fixed', or 'trainable'.
- **num_trainable_noise_layers** (int): The number of noise layers that should be trainable.
- **tile_latent** (bool): If `True`, the latent vector will be tiled across the batch; otherwise, each image will have its own latent vector.
- **bad_noise_layers** (str): A string of dot-separated indices indicating which noise layers should not be optimized.
- **opt_name** (str): The name of the optimizer to use. Options include 'sgd', 'adam', 'sgdm', and 'adamax'.
- **learning_rate** (float): The learning rate for the optimizer.
- **steps** (int): The number of optimization steps to perform.
- **lr_schedule** (str): The learning rate schedule to use. Options include 'fixed', 'linear1cycle', and 'linear1cycledrop'.
- **save_intermediate** (bool): If `True`, intermediate images will be yielded during optimization.
- **kwargs**: Additional keyword arguments for future extensions.

#### Return Value:
- Yields tuples of generated images and their downscaled versions during optimization if `save_intermediate` is `True`. If optimization completes, it yields the final generated image and its downscaled version.

#### Purpose:
Optimizes the latent variables and noise tensors to generate high-resolution images that match the reference images based on the specified loss function. The optimization process can be controlled through various parameters, including the optimizer type, learning rate, and number of steps.
```

