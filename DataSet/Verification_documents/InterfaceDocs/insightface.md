# API Documentation for FaceAnalysis Class

## Class: FaceAnalysis

### Description
The `FaceAnalysis` class is designed for face analysis, integrating functionalities such as face detection, keypoint recognition, and gender/age prediction. It utilizes ONNX models for various tasks related to face analysis.

### Attributes
- **models**: A dictionary that stores the loaded models for different tasks (e.g., detection, gender prediction).
- **model_dir**: The directory where the ONNX models are stored.
- **det_model**: The model specifically used for face detection.
- **det_thresh**: The threshold for face detection confidence.
- **det_size**: The size of the input image for detection.

### Method: `__init__`

#### Description
Initializes the `FaceAnalysis` class, loading the necessary models for face analysis.

#### Parameters
- **name** (str, optional): The name of the model to load. Default is `DEFAULT_MP_NAME`.
- **root** (str, optional): The root directory where models are stored. Default is `'~/.insightface'`.
- **allowed_modules** (list, optional): A list of allowed task names to filter the loaded models. Default is `None`.
- **kwargs**: Additional keyword arguments passed to the model loading function.

#### Return Value
None

---

### Method: `prepare`

#### Description
Prepares the models for inference by setting the detection threshold and input size.

#### Parameters
- **ctx_id** (int): The context ID for the execution provider (e.g., GPU or CPU).
- **det_thresh** (float, optional): The threshold for face detection confidence. Default is `0.5`. Must be in the range [0.0, 1.0].
- **det_size** (tuple, optional): The size of the input image for detection. Default is `(640, 640)`. Must be a tuple of two positive integers.

#### Return Value
None

---

### Method: `get`

#### Description
Detects faces in the provided image and retrieves face information, including bounding boxes and keypoints.

#### Parameters
- **img** (numpy.ndarray): The input image in which faces are to be detected. Must be a valid image array.
- **max_num** (int, optional): The maximum number of faces to detect. Default is `0`, which means no limit.
- **det_metric** (str, optional): The metric used for detection. Default is `'default'`.

#### Return Value
- **list**: A list of `Face` objects, each containing information about a detected face. If no faces are detected, returns an empty list.

---

### Method: `draw_on`

#### Description
Draws bounding boxes and keypoints on the detected faces in the image.

#### Parameters
- **img** (numpy.ndarray): The input image on which to draw the face annotations. Must be a valid image array.
- **faces** (list): A list of `Face` objects containing the detected face information.

#### Return Value
- **numpy.ndarray**: The image with drawn annotations (bounding boxes and keypoints).

---

### Example Usage
```python
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
rimg = app.draw_on(img, faces)
cv2.imwrite("./t1_output.jpg", rimg)
```

This documentation provides a comprehensive overview of the `FaceAnalysis` class and its methods, detailing their parameters, return values, and usage.