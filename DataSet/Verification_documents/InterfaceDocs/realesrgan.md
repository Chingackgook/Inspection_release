# API Documentation

## Class: RealESRGANer

The `RealESRGANer` class encapsulates the complete inference process of the RealESRGAN model, providing efficient, flexible, and robust image super-resolution capabilities, particularly suitable for handling large images and complex scenes.

### Attributes:
- **scale (int)**: The upsampling scale factor used in the networks. It is usually 2 or 4.
- **model_path (str)**: The path to the pretrained model. It can be a URL (will first download it automatically).
- **dni_weight (list, optional)**: Weights for Deep Network Interpolation. Default: None.
- **model (nn.Module, optional)**: The defined network. Default: None.
- **tile (int)**: If set, input images will be cropped into tiles for processing to avoid GPU memory issues. 0 denotes not using tiles. Default: 0.
- **tile_pad (int)**: The pad size for each tile to remove border artifacts. Default: 10.
- **pre_pad (int)**: Pad the input images to avoid border artifacts. Default: 10.
- **half (bool)**: Whether to use half precision during inference. Default: False.
- **device (torch.device)**: The device on which the model will run (CPU or GPU).
- **mod_scale (int)**: The modified scale for padding calculations.
- **img (torch.Tensor)**: The input image tensor after preprocessing.
- **output (torch.Tensor)**: The output image tensor after processing.

### Method: `__init__`

#### Parameters:
- **scale (int)**: The upsampling scale factor (2 or 4).
- **model_path (str)**: The path to the pretrained model.
- **dni_weight (list, optional)**: Weights for Deep Network Interpolation. Default: None.
- **model (nn.Module, optional)**: The defined network. Default: None.
- **tile (int)**: Tile size for processing. Default: 0.
- **tile_pad (int)**: Padding size for tiles. Default: 10.
- **pre_pad (int)**: Padding size for input images. Default: 10.
- **half (bool)**: Use half precision during inference. Default: False.
- **device (torch.device, optional)**: The device for model inference. Default: None.
- **gpu_id (int, optional)**: The GPU ID to use. Default: None.

#### Returns:
- None

#### Description:
Initializes the `RealESRGANer` class, setting up the model and device for inference.

---

### Method: `dni`

#### Parameters:
- **net_a (str)**: Path to the first network model.
- **net_b (str)**: Path to the second network model.
- **dni_weight (list)**: Weights for interpolation between the two networks.
- **key (str, optional)**: The key to access model parameters. Default: 'params'.
- **loc (str, optional)**: Location for loading models. Default: 'cpu'.

#### Returns:
- **dict**: The interpolated network parameters.

#### Description:
Performs Deep Network Interpolation to combine two models based on specified weights.

---

### Method: `pre_process`

#### Parameters:
- **img (numpy.ndarray)**: The input image to be pre-processed.

#### Returns:
- None

#### Description:
Pre-processes the input image by applying padding and ensuring it is divisible by the scale factor.

---

### Method: `process`

#### Parameters:
- None

#### Returns:
- None

#### Description:
Runs the model inference on the pre-processed image.

---

### Method: `tile_process`

#### Parameters:
- None

#### Returns:
- None

#### Description:
Processes the input image in tiles to manage memory usage, merging the results into a single output image.

---

### Method: `post_process`

#### Parameters:
- None

#### Returns:
- **torch.Tensor**: The final output image after removing padding.

#### Description:
Post-processes the output image by removing any extra padding applied during pre-processing.

---

### Method: `enhance`

#### Parameters:
- **img (numpy.ndarray)**: The input image to enhance.
- **outscale (float, optional)**: The output scale factor. Default: None.
- **alpha_upsampler (str, optional)**: Method for upsampling the alpha channel. Default: 'realesrgan'.

#### Returns:
- **numpy.ndarray**: The enhanced output image.
- **str**: The mode of the input image (e.g., 'RGB', 'L', 'RGBA').

#### Description:
Enhances the input image using the RealESRGAN model, optionally processing the alpha channel if present. The output can be scaled based on the specified `outscale`.