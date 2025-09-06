# API Documentation

## Class: LandmarksType
### Description
An enumeration class defining the types of landmarks that can be detected.

### Enum Values
- **TWO_D (1)**: Represents detected points `(x,y)` in a 2D space, following the visible contour of the face.
- **TWO_HALF_D (2)**: Represents the projection of 3D points into 2D.
- **THREE_D (3)**: Represents detected points `(x,y,z)` in a 3D space.

---

## Class: NetworkSize
### Description
An enumeration class defining the sizes of the neural network used for face alignment.

### Enum Values
- **LARGE (4)**: Represents the large network size.

---

## Class: FaceAlignment
### Description
A class for performing face alignment using deep learning models. It detects facial landmarks in images.

### Attributes
- **device (str)**: The device to run the model on (e.g., 'cuda' or 'cpu').
- **flip_input (bool)**: If True, the input image will be flipped horizontally before processing.
- **landmarks_type (LandmarksType)**: The type of landmarks to detect (2D, 2.5D, or 3D).
- **verbose (bool)**: If True, enables verbose output for debugging.
- **dtype (torch.dtype)**: The data type for the model (e.g., `torch.float32`).
- **face_detector**: An instance of the face detector used to locate faces in images.
- **face_alignment_net**: The neural network model for face alignment.
- **depth_prediciton_net**: The neural network model for depth prediction (only used for 3D landmarks).

### Method: __init__
#### Description
Initializes the FaceAlignment class with specified parameters.

#### Parameters
- **landmarks_type (LandmarksType)**: The type of landmarks to detect (must be one of the values from `LandmarksType`).
- **network_size (NetworkSize)**: The size of the network to use (default is `NetworkSize.LARGE`).
- **device (str)**: The device to run the model on (default is 'cuda').
- **dtype (torch.dtype)**: The data type for the model (default is `torch.float32`).
- **flip_input (bool)**: If True, the input image will be flipped horizontally before processing (default is False).
- **face_detector (str)**: The type of face detector to use (default is 'sfd').
- **face_detector_kwargs (dict)**: Additional keyword arguments for the face detector (default is None).
- **verbose (bool)**: If True, enables verbose output for debugging (default is False).

#### Return Value
None

---

### Method: get_landmarks
#### Description
Deprecated method to predict landmarks for faces in an image. Use `get_landmarks_from_image` instead.

#### Parameters
- **image_or_path (str or numpy.array or torch.tensor)**: The input image or path to the image.
- **detected_faces (list of numpy.array)**: List of bounding boxes for detected faces (default is None).
- **return_bboxes (bool)**: If True, return the face bounding boxes in addition to the keypoints (default is False).
- **return_landmark_score (bool)**: If True, return the keypoint scores along with the keypoints (default is False).

#### Return Value
- If both `return_bboxes` and `return_landmark_score` are False, returns landmarks.
- Otherwise, returns a tuple of (landmarks, landmark_scores, detected_faces).

---

### Method: get_landmarks_from_image
#### Description
Predicts the landmarks for each face present in the input image.

#### Parameters
- **image_or_path (str or numpy.array or torch.tensor)**: The input image or path to it.
- **detected_faces (list of numpy.array)**: List of bounding boxes for detected faces (default is None).
- **return_bboxes (bool)**: If True, return the face bounding boxes in addition to the keypoints (default is False).
- **return_landmark_score (bool)**: If True, return the keypoint scores along with the keypoints (default is False).

#### Return Value
- If both `return_bboxes` and `return_landmark_score` are False, returns landmarks.
- Otherwise, returns a tuple of (landmarks, landmark_scores, detected_faces).

---

### Method: get_landmarks_from_batch
#### Description
Predicts the landmarks for each face present in a batch of images in parallel.

#### Parameters
- **image_batch (torch.tensor)**: The input images batch.
- **detected_faces (list of numpy.array)**: List of bounding boxes for detected faces (default is None).
- **return_bboxes (bool)**: If True, return the face bounding boxes in addition to the keypoints (default is False).
- **return_landmark_score (bool)**: If True, return the keypoint scores along with the keypoints (default is False).

#### Return Value
- If both `return_bboxes` and `return_landmark_score` are False, returns landmarks.
- Otherwise, returns a tuple of (landmarks, landmarks_scores_list, detected_faces).

---

### Method: get_landmarks_from_directory
#### Description
Scans a directory for images and predicts the landmarks for each face present in the images found.

#### Parameters
- **path (str)**: Path to the target directory containing the images.
- **extensions (list of str)**: List of image extensions considered (default is ['.jpg', '.png']).
- **recursive (bool)**: If True, scans for images recursively (default is True).
- **show_progress_bar (bool)**: If True, displays a progress bar (default is True).
- **return_bboxes (bool)**: If True, return the face bounding boxes in addition to the keypoints (default is False).
- **return_landmark_score (bool)**: If True, return the keypoint scores along with the keypoints (default is False).

#### Return Value
A dictionary where keys are image paths and values are the predicted landmarks, bounding boxes, and scores (if requested).

