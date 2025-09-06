# API Documentation

## Class: `TracedModel`

The `TracedModel` class is a wrapper for the YOLOR model that converts it into a traced model using PyTorch's JIT compilation. This class is designed to optimize the model for inference by tracing its operations.

### Attributes:
- `model`: The original YOLOR model that is being wrapped and traced.
- `stride`: The stride of the model, which is used in the detection process.
- `names`: The names of the classes that the model can detect.
- `detect_layer`: The final detection layer of the model, which processes the output of the traced model.

### Method: `__init__`

```python
def __init__(self, model=None, device=None, img_size=(640, 640)):
```

#### Parameters:
- `model` (nn.Module): The YOLOR model to be wrapped and traced. This parameter is required and should be an instance of a PyTorch model.
- `device` (str or torch.device): The device on which the model will be loaded (e.g., 'cpu' or 'cuda'). This parameter is required.
- `img_size` (tuple): The size of the input images as a tuple (height, width). Default is (640, 640). The values should be positive integers.

#### Return Value:
- None

#### Purpose:
Initializes the `TracedModel` instance by converting the provided YOLOR model into a traced model using a random input tensor. The traced model is saved to a file named "traced_model.pt". The model is set to evaluation mode and moved to the specified device.

---

### Method: `forward`

```python
def forward(self, x, augment=False, profile=False):
```

#### Parameters:
- `x` (torch.Tensor): The input tensor containing the images to be processed. The shape should be (N, 3, H, W), where N is the batch size, 3 is the number of color channels, and H and W are the height and width of the images, respectively. H and W should match the `img_size` specified during initialization.
- `augment` (bool): A flag indicating whether to apply augmentation during inference. Default is `False`. This parameter can be set to `True` to enable augmentations.
- `profile` (bool): A flag indicating whether to profile the model's performance. Default is `False`. This parameter can be set to `True` to enable profiling.

#### Return Value:
- `out`: The output of the model after processing the input tensor. The output format depends on the specific implementation of the detection layer.

#### Purpose:
Processes the input tensor through the traced model and the detection layer, returning the model's output. This method is called during inference to obtain predictions from the model.

