```markdown
# API Documentation

## Class: OcclusionAwareGenerator

### Description
The `OcclusionAwareGenerator` class is designed to generate animations by applying movements from driving videos to source images. It utilizes keypoint detection to align the source and driving frames, enabling realistic animation generation. The generator follows the Johnson architecture and can estimate occlusion maps.

### Attributes
- **num_channels** (int): The number of channels in the input images (e.g., 3 for RGB images).
- **num_kp** (int): The number of keypoints used for driving the animation.
- **block_expansion** (int): The expansion factor for the number of features in the network.
- **max_features** (int): The maximum number of features in the network.
- **num_down_blocks** (int): The number of downsampling blocks in the network.
- **num_bottleneck_blocks** (int): The number of bottleneck blocks in the network.
- **estimate_occlusion_map** (bool): A flag indicating whether to estimate an occlusion map.
- **dense_motion_network** (DenseMotionNetwork or None): An instance of the DenseMotionNetwork for estimating dense motion.
- **estimate_jacobian** (bool): A flag indicating whether to estimate the Jacobian.
- **first** (SameBlock2d): The first block of the generator.
- **down_blocks** (ModuleList): A list of downsampling blocks.
- **up_blocks** (ModuleList): A list of upsampling blocks.
- **bottleneck** (Sequential): A sequential container for bottleneck blocks.
- **final** (Conv2d): The final convolutional layer of the generator.

### Methods

#### __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False)

**Parameters:**
- **num_channels** (int): Number of channels in the input images (e.g., 3 for RGB).
- **num_kp** (int): Number of keypoints for driving the animation.
- **block_expansion** (int): Expansion factor for the number of features.
- **max_features** (int): Maximum number of features in the network.
- **num_down_blocks** (int): Number of downsampling blocks.
- **num_bottleneck_blocks** (int): Number of bottleneck blocks.
- **estimate_occlusion_map** (bool, optional): Whether to estimate an occlusion map (default is False).
- **dense_motion_params** (dict or None, optional): Parameters for the dense motion network (default is None).
- **estimate_jacobian** (bool, optional): Whether to estimate the Jacobian (default is False).

**Returns:** 
- None

**Description:** 
Initializes the `OcclusionAwareGenerator` class, setting up the network architecture and parameters.

---

#### deform_input(self, inp, deformation)

**Parameters:**
- **inp** (Tensor): The input tensor to be deformed, with shape (N, C, H, W).
- **deformation** (Tensor): The deformation field, with shape (N, H_old, W_old, 2).

**Returns:** 
- **Tensor**: The deformed input tensor.

**Description:** 
Applies a deformation to the input tensor using the provided deformation field. If the input tensor's dimensions do not match the deformation field's dimensions, it resizes the deformation field accordingly.

---

#### forward(self, source_image, kp_driving, kp_source)

**Parameters:**
- **source_image** (Tensor): The source image tensor, with shape (N, C, H, W).
- **kp_driving** (Tensor): The keypoints for the driving video, with shape (N, num_kp, 2).
- **kp_source** (Tensor): The keypoints for the source image, with shape (N, num_kp, 2).

**Returns:** 
- **dict**: A dictionary containing:
  - **'mask'** (Tensor): The mask indicating valid regions after deformation.
  - **'sparse_deformed'** (Tensor): The sparse deformed representation of the source image.
  - **'occlusion_map'** (Tensor, optional): The estimated occlusion map, if applicable.
  - **'deformed'** (Tensor): The deformed version of the source image.
  - **'prediction'** (Tensor): The final generated output image.

**Description:** 
Processes the source image and keypoints to generate a deformed output image based on the driving video. It includes encoding, transforming, and decoding steps, while also estimating occlusion if required.
```