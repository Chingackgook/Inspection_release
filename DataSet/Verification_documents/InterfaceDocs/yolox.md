# API Documentation

## Function: `vis`

### Description
The `vis` function visualizes bounding boxes on an image, along with class labels and confidence scores. It draws rectangles around detected objects and annotates them with text.

### Parameters
- **img** (`numpy.ndarray`): 
  - Description: The input image on which the bounding boxes will be drawn.
  - Value Range: Should be a valid image array (height x width x channels).
  
- **boxes** (`list` of `numpy.ndarray`): 
  - Description: A list of bounding boxes, where each box is represented as an array of four coordinates `[x0, y0, x1, y1]`.
  - Value Range: Each box should contain four integers representing the top-left and bottom-right corners of the rectangle.

- **scores** (`list` of `float`): 
  - Description: A list of confidence scores corresponding to each bounding box.
  - Value Range: Each score should be a float between 0 and 1.

- **cls_ids** (`list` of `int`): 
  - Description: A list of class IDs corresponding to each bounding box.
  - Value Range: Each ID should be a non-negative integer that maps to a class in `class_names`.

- **conf** (`float`, optional): 
  - Description: The confidence threshold for displaying bounding boxes. Boxes with scores below this threshold will not be drawn.
  - Value Range: Should be a float between 0 and 1 (default is 0.5).

- **class_names** (`list` of `str`, optional): 
  - Description: A list of class names corresponding to the class IDs. Used for labeling the bounding boxes.
  - Value Range: Should contain strings representing class names.

### Returns
- **numpy.ndarray**: 
  - Description: The input image with drawn bounding boxes and annotated text.

### Example Usage
```python
import cv2
import numpy as np

# Example image
image = cv2.imread('image.jpg')

# Example bounding boxes, scores, and class IDs
boxes = np.array([[50, 50, 200, 200], [30, 30, 100, 100]])
scores = [0.9, 0.4]
cls_ids = [0, 1]
class_names = ['cat', 'dog']

# Visualize
output_image = vis(image, boxes, scores, cls_ids, conf=0.5, class_names=class_names)

# Display the result
cv2.imshow('Output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Purpose
The `vis` function is designed to facilitate the visualization of object detection results by overlaying bounding boxes and labels on images, making it easier to interpret the output of detection algorithms.

