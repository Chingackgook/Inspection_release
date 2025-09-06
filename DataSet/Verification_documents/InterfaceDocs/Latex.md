# API Documentation for LatexOCR

## Class: `LatexOCR`

The `LatexOCR` class provides a simple interface to predict LaTeX code from images using a trained model. It includes methods for initializing the model and making predictions.

### Attributes:
- **image_resizer**: (Optional) A model used to resize images if necessary.
- **last_pic**: (Optional) Stores the last processed image for reuse.

### Method: `__init__(self, arguments=None)`

#### Description:
Initializes a `LatexOCR` model with specified parameters or default settings.

#### Parameters:
- **arguments** (Union[Namespace, Munch], optional): Special model parameters. If not provided, defaults to a predefined configuration.
  - **config** (str): Path to the configuration YAML file. Default is 'settings/config.yaml'.
  - **checkpoint** (str): Path to the model weights file. Default is 'checkpoints/weights.pth'.
  - **no_cuda** (bool): If True, disables CUDA. Default is False.
  - **no_resize** (bool): If True, disables image resizing. Default is False.

#### Return Value:
None

#### Purpose:
To set up the model with the necessary configurations and load the required weights for making predictions.

---

### Method: `__call__(self, img=None, resize=True)`

#### Description:
Generates a prediction of LaTeX code from a given image.

#### Parameters:
- **img** (Image, optional): The image to predict. If None, the last processed image is used. Default is None.
- **resize** (bool, optional): Indicates whether to use the image resizer model. Default is True.

#### Return Value:
- **str**: The predicted LaTeX code.

#### Purpose:
To process the input image and return the corresponding LaTeX code prediction. If no image is provided, it uses the last processed image. The method also handles resizing if specified.

#### Example Usage:
```python
latex_ocr = LatexOCR()
predicted_latex = latex_ocr(image)
```

---

This documentation provides a clear understanding of the `LatexOCR` class, its initializer, and its call method, including their parameters, return values, and purposes.

