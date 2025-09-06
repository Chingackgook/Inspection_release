```markdown
# API Documentation

## Class: MattingNetwork

The `MattingNetwork` class implements a deep learning model for image matting, which is the process of extracting foreground objects from images. It utilizes different backbone architectures and refiner methods to achieve high-quality results.

### Attributes:
- **backbone**: A backbone network for feature extraction. Can be either MobileNetV3 or ResNet50.
- **aspp**: Atrous Spatial Pyramid Pooling module for multi-scale feature extraction.
- **decoder**: A recurrent decoder that reconstructs the output from the features extracted by the backbone.
- **project_mat**: A projection layer that outputs the foreground residual and alpha matte.
- **project_seg**: A projection layer that outputs the segmentation map.
- **refiner**: A refiner module that enhances the quality of the output, either using a fast guided filter or a deep guided filter.

### Method: `__init__`

```python
def __init__(self, variant: str = 'mobilenetv3', refiner: str = 'deep_guided_filter', pretrained_backbone: bool = False):
```

#### Parameters:
- **variant** (`str`): The backbone architecture to use. Options are:
  - `'mobilenetv3'`: Use MobileNetV3 as the backbone.
  - `'resnet50'`: Use ResNet50 as the backbone.
  
- **refiner** (`str`): The type of refiner to use. Options are:
  - `'fast_guided_filter'`: Use a fast guided filter for refinement.
  - `'deep_guided_filter'`: Use a deep guided filter for refinement.
  
- **pretrained_backbone** (`bool`): If `True`, initializes the backbone with pretrained weights. Default is `False`.

#### Return Value:
- None

#### Purpose:
Initializes the `MattingNetwork` class, setting up the backbone, ASPP, decoder, projection layers, and refiner based on the specified parameters.

---

### Method: `forward`

```python
def forward(self, src: Tensor, r1: Optional[Tensor] = None, r2: Optional[Tensor] = None, r3: Optional[Tensor] = None, r4: Optional[Tensor] = None, downsample_ratio: float = 1, segmentation_pass: bool = False):
```

#### Parameters:
- **src** (`Tensor`): The input image tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels, and H and W are the height and width of the image.
  
- **r1**, **r2**, **r3**, **r4** (`Optional[Tensor]`): Optional residual tensors from previous decoder steps. These can be used for recurrent decoding.

- **downsample_ratio** (`float`): The ratio by which to downsample the input image. Default is `1` (no downsampling). Must be greater than `0`.

- **segmentation_pass** (`bool`): If `True`, the model will output segmentation maps instead of the foreground and alpha matte. Default is `False`.

#### Return Value:
- Returns a list containing:
  - If `segmentation_pass` is `False`: 
    - `fgr` (`Tensor`): The foreground image tensor.
    - `pha` (`Tensor`): The alpha matte tensor.
    - `rec` (`List[Tensor]`): A list of recurrent decoder outputs.
  - If `segmentation_pass` is `True`: 
    - `seg` (`Tensor`): The segmentation map tensor.
    - `rec` (`List[Tensor]`): A list of recurrent decoder outputs.

#### Purpose:
Performs a forward pass through the network, processing the input image to produce either the foreground image and alpha matte or the segmentation map, depending on the `segmentation_pass` flag.
```

