# API Documentation for FaceRestoreHelper

## Class: FaceRestoreHelper
The `FaceRestoreHelper` class encapsulates the complete processing pipeline for face restoration, managing the entire workflow from face detection to final image restoration. It implements a standardized process for face detection, alignment, restoration, and blending, utilizing multiple models and adaptive parameter configurations to achieve high-quality results.

### Attributes:
- `upscale_factor`: (int) The factor by which to upscale the image.
- `face_size`: (tuple) The size of the face to be processed, defined as (width, height).
- `crop_ratio`: (tuple) The ratio for cropping the face, defined as (height_ratio, width_ratio).
- `det_model`: (str) The model used for face detection (e.g., 'retinaface_resnet50', 'dlib').
- `save_ext`: (str) The file extension for saving images (e.g., 'png').
- `template_3points`: (bool) Flag to use a 3-point template for face alignment.
- `pad_blur`: (bool) Flag to enable padding and blurring for images.
- `use_parse`: (bool) Flag to enable face parsing.
- `device`: (torch.device) The device to run the model on (CPU or GPU).
- `all_landmarks_5`: (list) List to store detected landmarks for faces.
- `det_faces`: (list) List to store detected face bounding boxes.
- `affine_matrices`: (list) List to store affine transformation matrices for alignment.
- `inverse_affine_matrices`: (list) List to store inverse affine transformation matrices.
- `cropped_faces`: (list) List to store cropped face images.
- `restored_faces`: (list) List to store restored face images.
- `pad_input_imgs`: (list) List to store padded input images.

### Method: `__init__(self, upscale_factor, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', template_3points=False, pad_blur=False, use_parse=False, device=None)`
#### Parameters:
- `upscale_factor` (int): The factor by which to upscale the image. Must be greater than or equal to 1.
- `face_size` (int, optional): The size of the face to be processed. Default is 512.
- `crop_ratio` (tuple, optional): The ratio for cropping the face, defined as (height_ratio, width_ratio). Both values must be greater than or equal to 1.
- `det_model` (str, optional): The model used for face detection. Default is 'retinaface_resnet50'.
- `save_ext` (str, optional): The file extension for saving images. Default is 'png'.
- `template_3points` (bool, optional): Flag to use a 3-point template for face alignment. Default is False.
- `pad_blur` (bool, optional): Flag to enable padding and blurring for images. Default is False.
- `use_parse` (bool, optional): Flag to enable face parsing. Default is False.
- `device` (torch.device, optional): The device to run the model on (CPU or GPU). Default is None, which automatically selects the device.

#### Returns:
None

#### Description:
Initializes the `FaceRestoreHelper` class, setting up the necessary parameters and initializing the face detection and parsing models.

---

### Method: `set_upscale_factor(self, upscale_factor)`
#### Parameters:
- `upscale_factor` (int): The new factor by which to upscale the image. Must be greater than or equal to 1.

#### Returns:
None

#### Description:
Sets the upscale factor for the image processing pipeline.

---

### Method: `read_image(self, img)`
#### Parameters:
- `img` (str or numpy.ndarray): The image to be read, which can be a file path or a loaded image in BGR format.

#### Returns:
None

#### Description:
Reads an image from a file path or a loaded image, processes it, and prepares it for face detection.

---

### Method: `init_dlib(self, detection_path, landmark5_path)`
#### Parameters:
- `detection_path` (str): The file path for the dlib face detection model.
- `landmark5_path` (str): The file path for the dlib landmark predictor model.

#### Returns:
- `(face_detector, shape_predictor_5)`: A tuple containing the initialized face detector and shape predictor.

#### Description:
Initializes the dlib face detection and landmark prediction models.

---

### Method: `get_face_landmarks_5_dlib(self, only_keep_largest=False, scale=1)`
#### Parameters:
- `only_keep_largest` (bool, optional): If True, only keeps the largest detected face. Default is False.
- `scale` (float, optional): The scale factor for face detection. Default is 1.

#### Returns:
- (int): The number of detected faces.

#### Description:
Detects face landmarks using the dlib model and stores them for further processing.

---

### Method: `get_face_landmarks_5(self, only_keep_largest=False, only_center_face=False, resize=None, blur_ratio=0.01, eye_dist_threshold=None)`
#### Parameters:
- `only_keep_largest` (bool, optional): If True, only keeps the largest detected face. Default is False.
- `only_center_face` (bool, optional): If True, only keeps the center face. Default is False.
- `resize` (int, optional): The size to which the image should be resized before detection. Default is None.
- `blur_ratio` (float, optional): The ratio for blurring the padded images. Default is 0.01.
- `eye_dist_threshold` (float, optional): The minimum eye distance threshold for face detection. Default is None.

#### Returns:
- (int): The number of detected faces.

#### Description:
Detects face landmarks using the specified detection model and stores them for further processing.

---

### Method: `align_warp_face(self, save_cropped_path=None, border_mode='constant')`
#### Parameters:
- `save_cropped_path` (str, optional): The path to save the cropped face images. Default is None.
- `border_mode` (str, optional): The border mode for image warping. Options include 'constant', 'reflect101', and 'reflect'. Default is 'constant'.

#### Returns:
None

#### Description:
Aligns and warps detected faces using the specified face template and saves the cropped faces if a path is provided.

---

### Method: `get_inverse_affine(self, save_inverse_affine_path=None)`
#### Parameters:
- `save_inverse_affine_path` (str, optional): The path to save the inverse affine matrices. Default is None.

#### Returns:
None

#### Description:
Calculates and stores the inverse affine matrices for the aligned faces, saving them if a path is provided.

---

### Method: `add_restored_face(self, restored_face, input_face=None)`
#### Parameters:
- `restored_face` (numpy.ndarray): The restored face image to be added.
- `input_face` (numpy.ndarray, optional): The original input face image for color transfer. Default is None.

#### Returns:
None

#### Description:
Adds a restored face image to the list of restored faces, optionally transferring color from the input face if provided.

---

### Method: `paste_faces_to_input_image(self, save_path=None, upsample_img=None, draw_box=False, face_upsampler=None)`
#### Parameters:
- `save_path` (str, optional): The path to save the final image with pasted faces. Default is None.
- `upsample_img` (numpy.ndarray, optional): An optional image to use as the background. Default is None.
- `draw_box` (bool, optional): If True, draws bounding boxes around pasted faces. Default is False.
- `face_upsampler` (object, optional): An optional face upsampler object for enhancing the restored faces. Default is None.

#### Returns:
- (numpy.ndarray): The final image with pasted faces.

#### Description:
Pastes the restored faces back onto the input image, blending them seamlessly and saving the result if a path is provided.

---

### Method: `clean_all(self)`
#### Parameters:
None

#### Returns:
None

#### Description:
Clears all stored landmarks, restored faces, affine matrices, cropped faces, and detected faces, resetting the state of the `FaceRestoreHelper` instance.