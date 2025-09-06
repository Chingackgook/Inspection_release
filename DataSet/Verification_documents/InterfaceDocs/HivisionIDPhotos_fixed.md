# API Documentation

## Class: IDCreator

### Description
The `IDCreator` class is designed to handle the complete process of generating ID photos, including image matting, beauty enhancement, face detection, and image adjustments.

### Attributes
- **before_all**: `ContextHandler`
  - Description: A callback that is executed before all processing steps. It is called when the image has been resized to a maximum edge length of 2000 pixels.
  
- **after_matting**: `ContextHandler`
  - Description: A callback that is executed after the matting process. At this point, `ctx.matting_image` is assigned.

- **after_detect**: `ContextHandler`
  - Description: A callback that is executed after face detection. At this point, `ctx.face` is assigned. This callback is not executed if only background change is performed.

- **after_all**: `ContextHandler`
  - Description: A callback that is executed after all processing steps. At this point, `ctx.result` is assigned.

- **matting_handler**: `ContextHandler`
  - Description: The handler responsible for the human matting process.

- **detection_handler**: `ContextHandler`
  - Description: The handler responsible for face detection.

- **beauty_handler**: `ContextHandler`
  - Description: The handler responsible for beauty enhancement.

- **ctx**: `Context`
  - Description: The context object that holds parameters and intermediate results during processing.

### Method: `__init__`

#### Description
Initializes an instance of the `IDCreator` class, setting up the necessary attributes and handlers for processing ID photos.

#### Parameters
- No parameters.

#### Return Value
- Returns an instance of the `IDCreator` class.

---

### Method: `__call__`

#### Description
Processes the input image to generate an ID photo, applying matting, beauty enhancements, face detection, and adjustments as specified by the parameters.

#### Parameters
- **image**: `np.ndarray`
  - Description: The input image to be processed.
  
- **size**: `Tuple[int, int]`, default `(413, 295)`
  - Description: The desired output image size (height, width).
  
- **change_bg_only**: `bool`, default `False`
  - Description: If `True`, only the background will be changed without further processing.
  
- **crop_only**: `bool`, default `False`
  - Description: If `True`, only cropping will be performed without matting.
  
- **head_measure_ratio**: `float`, default `0.2`
  - Description: The expected ratio of the face area to the total image area.
  
- **head_height_ratio**: `float`, default `0.45`
  - Description: The expected ratio of the face center's height in the total image height.
  
- **head_top_range**: `Tuple[float, float]`, default `(0.12, 0.1)`
  - Description: The range for the head's distance from the top of the image (max, min).
  
- **face**: `Tuple[int, int, int, int]`, default `None`
  - Description: The coordinates of the face in the image (x1, y1, x2, y2).
  
- **whitening_strength**: `int`, default `0`
  - Description: The strength of the whitening effect to be applied.
  
- **brightness_strength**: `int`, default `0`
  - Description: The strength of the brightness adjustment.
  
- **contrast_strength**: `int`, default `0`
  - Description: The strength of the contrast adjustment.
  
- **sharpen_strength**: `int`, default `0`
  - Description: The strength of the sharpening effect.
  
- **saturation_strength**: `int`, default `0`
  - Description: The strength of the saturation adjustment.
  
- **face_alignment**: `bool`, default `False`
  - Description: If `True`, face alignment will be performed if the roll angle exceeds 2 degrees.

#### Return Value
- Returns an instance of `Result`, which contains:
  - `standard`: The processed standard ID photo.
  - `hd`: The processed high-definition ID photo.
  - `matting`: The image after matting.
  - `clothing_params`: Parameters related to clothing adjustments.
  - `typography_params`: Parameters related to typography adjustments.
  - `face`: The detected face information.

#### Purpose
This method encapsulates the entire ID photo generation process, allowing users to specify various parameters for customization and control over the output.