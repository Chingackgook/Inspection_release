# API Documentation for Fawkes

## Class: Fawkes

The `Fawkes` class is designed to provide functionality for generating cloaked images that protect the identity of individuals in photographs. It utilizes a feature extraction method to create masks that obscure facial features while maintaining the overall appearance of the image.

### Attributes:
- **feature_extractor**: The feature extractor model used for generating masks.
- **gpu**: The GPU device to be used for processing. If `None`, CPU will be used.
- **batch_size**: The number of images to process in a single batch.
- **mode**: The mode of operation, which determines the parameters for mask generation.
- **th**: Threshold value for mask generation, determined by the mode.
- **lr**: Learning rate for the optimization process, determined by the mode.
- **max_step**: Maximum number of iterations for the optimization process, determined by the mode.
- **protector**: An instance of `FawkesMaskGeneration` used for generating cloaked images.
- **protector_param**: A string representing the current parameters for the protector.
- **feature_extractors_ls**: A list of feature extractor models loaded based on the selected mode.
- **aligner**: An instance of the face aligner used for aligning faces in images.

### Method: `__init__`

```python
def __init__(self, feature_extractor, gpu, batch_size, mode="low"):
```

#### Parameters:
- **feature_extractor**: (str) The name of the feature extractor to be used.
- **gpu**: (str or None) The GPU device to be used for processing. If `None`, CPU will be used.
- **batch_size**: (int) The number of images to process in a single batch. Must be greater than 0.
- **mode**: (str) The mode of operation. Must be one of 'low', 'mid', or 'high'. Default is 'low'.

#### Return Value:
- None

#### Purpose:
Initializes the `Fawkes` class with the specified feature extractor, GPU settings, batch size, and mode. It sets up the necessary parameters for mask generation based on the selected mode.

---

### Method: `mode2param`

```python
def mode2param(self, mode):
```

#### Parameters:
- **mode**: (str) The mode of operation. Must be one of 'low', 'mid', or 'high'.

#### Return Value:
- (tuple) A tuple containing:
  - **th**: (float) The threshold value for mask generation.
  - **max_step**: (int) The maximum number of iterations for the optimization process.
  - **lr**: (int) The learning rate for the optimization process.
  - **extractors**: (list) A list of feature extractor names to be used.

#### Purpose:
Maps the specified mode to its corresponding parameters for mask generation, including threshold, maximum steps, learning rate, and feature extractors.

---

### Method: `run_protection`

```python
def run_protection(self, image_paths, th=0.04, sd=1e7, lr=10, max_step=500, batch_size=1, format='png',
                   separate_target=True, debug=False, no_align=False, exp="", maximize=True,
                   save_last_on_failed=True):
```

#### Parameters:
- **image_paths**: (list of str) A list of file paths to the images to be processed.
- **th**: (float) The threshold value for mask generation. Default is 0.04.
- **sd**: (float) The initial constant for the optimization process. Default is 1e7.
- **lr**: (int) The learning rate for the optimization process. Default is 10.
- **max_step**: (int) The maximum number of iterations for the optimization process. Default is 500.
- **batch_size**: (int) The number of images to process in a single batch. Default is 1.
- **format**: (str) The format to save the output images. Default is 'png'.
- **separate_target**: (bool) Whether to separate the target images. Default is True.
- **debug**: (bool) Whether to enable debug mode. Default is False.
- **no_align**: (bool) Whether to skip face alignment. Default is False.
- **exp**: (str) An optional experiment identifier. Default is an empty string.
- **maximize**: (bool) Whether to maximize the loss function during optimization. Default is True.
- **save_last_on_failed**: (bool) Whether to save the last image on failure. Default is True.

#### Return Value:
- (int) Returns 1 if the process is successful, 2 if no faces are detected, and 3 if no images are found.

#### Purpose:
Processes the specified images to generate cloaked versions that protect the identities of individuals. It performs face detection, mask generation, and saves the resulting images in the specified format.

