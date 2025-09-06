# API Documentation for MagicAnimate

## Class: `MagicAnimate`

The `MagicAnimate` class provides an interface for generating animations based on a source image and a motion sequence. It utilizes various models and pipelines to create animated outputs.

### Attributes:
- `pipeline`: An instance of `AnimationPipeline` that combines various models for generating animations.
- `appearance_encoder`: An instance of `AppearanceEncoderModel` used for encoding the appearance of the source image.
- `reference_control_writer`: An instance of `ReferenceAttentionControl` for writing reference controls.
- `reference_control_reader`: An instance of `ReferenceAttentionControl` for reading reference controls.
- `L`: An integer representing the length of the video sequence.

### Method: `__init__`

```python
def __init__(self, config="configs/prompts/animation.yaml") -> None:
```

#### Parameters:
- `config` (str): The path to the configuration file (default is `"configs/prompts/animation.yaml"`). This file contains settings for the animation pipeline.

#### Return Value:
- None

#### Description:
Initializes the `MagicAnimate` class by loading the necessary models and configurations. It sets up the animation pipeline and prepares the models for inference.

---

### Method: `__call__`

```python
def __call__(self, source_image, motion_sequence, random_seed, step, guidance_scale, size=512):
```

#### Parameters:
- `source_image` (numpy.ndarray): A 3D array representing the source image to be animated. The shape should be `(H, W, C)` where `H` is height, `W` is width, and `C` is the number of color channels (typically 3 for RGB).
- `motion_sequence` (str): The path to a video file (e.g., `.mp4`) that contains the motion sequence to be applied to the source image.
- `random_seed` (int): A seed for random number generation. This allows for reproducibility of results. If set to `-1`, a random seed will be generated.
- `step` (int): The number of inference steps to be used during the animation generation. This should be a positive integer.
- `guidance_scale` (float): A scaling factor for guidance during the animation generation. This should be a positive float.
- `size` (int, optional): The size (height and width) to which the source image and motion sequence frames will be resized. Default is `512`. 

#### Return Value:
- `animation_path` (str): The file path to the generated animation video.

#### Description:
Generates an animation based on the provided source image and motion sequence. It processes the input, applies the animation pipeline, and saves the output video to a specified directory. The method handles resizing of images and videos, manages random seed settings for reproducibility, and returns the path to the saved animation.

---

### Example Usage:
```python
# Initialize the MagicAnimate class
magic_animate = MagicAnimate(config="path/to/config.yaml")

# Call the instance with the required parameters
animation_path = magic_animate(source_image, motion_sequence, random_seed=42, step=50, guidance_scale=7.5, size=512)
```

This documentation provides a comprehensive overview of the `MagicAnimate` class, its initializer, and the `__call__` method, detailing their parameters, return values, and purposes.

