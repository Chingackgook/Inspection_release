# API Documentation for DPT Class

## Class: DPT

The `DPT` class is designed for depth and normal estimation from images using a pretrained model. It supports two tasks: depth estimation and normal estimation.

### Attributes:
- **task (str)**: Specifies the task to perform. Acceptable values are `'depth'` for depth estimation and `'normal'` for normal estimation.
- **device (str)**: Specifies the device to run the model on. Acceptable values are `'cuda'` for GPU or `'cpu'` for CPU.
- **model (DPTDepthModel)**: The model instance used for depth or normal estimation.
- **aug (torchvision.transforms.Compose)**: A composition of image transformations applied to the input image before feeding it to the model.

### Method: `__init__`

```python
def __init__(self, task='depth', device='cuda'):
```

#### Parameters:
- **task (str)**: The task to perform. Default is `'depth'`. Acceptable values are:
  - `'depth'`: For depth estimation.
  - `'normal'`: For normal estimation.
  
- **device (str)**: The device to run the model on. Default is `'cuda'`. Acceptable values are:
  - `'cuda'`: Use GPU for computation.
  - `'cpu'`: Use CPU for computation.

#### Return Value:
- None

#### Description:
Initializes the DPT class by loading the appropriate model based on the specified task and device. It also sets up the necessary image transformations for preprocessing.

---

### Method: `__call__`

```python
def __call__(self, image):
```

#### Parameters:
- **image (np.ndarray)**: The input image to be processed. It should be a NumPy array of shape `[H, W, 3]` where:
  - `H` is the height of the image.
  - `W` is the width of the image.
  - The array should contain pixel values in the range of `[0, 255]` (uint8 format).

#### Return Value:
- **depth (np.ndarray)**: If the task is `'depth'`, returns a NumPy array of shape `[H, W]` representing the estimated depth map, with values clamped between `0` and `1`.
- **normal (np.ndarray)**: If the task is `'normal'`, returns a NumPy array of shape `[H, W, 3]` representing the estimated normal map, with values clamped between `0` and `1`.

#### Description:
Processes the input image to produce either a depth map or a normal map based on the specified task. The image is transformed, passed through the model, and the output is resized to match the original image dimensions.

--- 

This documentation provides a clear understanding of the `DPT` class, its initialization, and how to use it for image processing tasks related to depth and normal estimation.

