# API Documentation for DeepFace

## 1. `build_model`

### Description
Builds a pre-trained model for various tasks related to facial recognition and analysis.

### Parameters
- **model_name** (str): Identifier for the model to be built. Options include:
  - For face recognition: `VGG-Face`, `Facenet`, `Facenet512`, `OpenFace`, `DeepFace`, `DeepID`, `Dlib`, `ArcFace`, `SFace`, `GhostFaceNet`, `Buffalo_L`.
  - For facial attributes: `Age`, `Gender`, `Emotion`, `Race`.
  - For face detectors: `opencv`, `mtcnn`, `ssd`, `dlib`, `retinaface`, `mediapipe`, `yolov8`, `yolov11n`, `yolov11s`, `yolov11m`, `yunet`, `fastmtcnn`, `centerface`.
  - For spoofing: `Fasnet`.
  
- **task** (str): The task for which the model is built. Default is `"facial_recognition"`. Options include:
  - `"facial_recognition"`
  - `"facial_attribute"`
  - `"face_detector"`
  - `"spoofing"`

### Returns
- **Any**: The built model for the specified task.

---

## 2. `verify`

### Description
Verifies if two images represent the same person or different persons.

### Parameters
- **img1_path** (Union[str, np.ndarray, IO[bytes], List[float]]): Path to the first image.
- **img2_path** (Union[str, np.ndarray, IO[bytes], List[float]]): Path to the second image.
- **model_name** (str): Model for face recognition. Default is `"VGG-Face"`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`. Options include:
  - `'opencv'`, `'retinaface'`, `'mtcnn'`, `'ssd'`, `'dlib'`, `'mediapipe'`, `'yolov8'`, `'yolov11n'`, `'yolov11s'`, `'yolov11m'`, `'centerface'`, `'skip'`.
- **distance_metric** (str): Metric for measuring similarity. Default is `"cosine"`. Options include:
  - `'cosine'`, `'euclidean'`, `'euclidean_l2'`, `'angular'`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **align** (bool): Flag to enable face alignment. Default is `True`.
- **expand_percentage** (int): Expand detected facial area with a percentage. Default is `0`.
- **normalization** (str): Normalize the input image. Default is `"base"`.
- **silent** (bool): Suppress log messages. Default is `False`.
- **threshold** (Optional[float]): Threshold for determining if images represent the same person. Default is `None`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.

### Returns
- **Dict[str, Any]**: A dictionary containing verification results, including:
  - `'verified'` (bool): Indicates if the images represent the same person.
  - `'distance'` (float): Distance measure between face vectors.
  - `'threshold'` (float): Maximum threshold used for verification.
  - `'model'` (str): The chosen face recognition model.
  - `'distance_metric'` (str): The chosen similarity metric.
  - `'facial_areas'` (dict): Rectangular regions of interest for faces in both images.
  - `'time'` (float): Time taken for the verification process in seconds.

---

## 3. `analyze`

### Description
Analyzes facial attributes such as age, gender, emotion, and race in the provided image.

### Parameters
- **img_path** (Union[str, np.ndarray, IO[bytes], List[str], List[np.ndarray], List[IO[bytes]]]): The exact path to the image or a numpy array.
- **actions** (Union[tuple, list]): Attributes to analyze. Default is `("emotion", "age", "gender", "race")`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **align** (bool): Perform alignment based on eye positions. Default is `True`.
- **expand_percentage** (int): Expand detected facial area with a percentage. Default is `0`.
- **silent** (bool): Suppress log messages. Default is `False`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.

### Returns
- **Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]:** A list of analysis results for each detected face, including:
  - `'region'` (dict): Rectangular region of the detected face.
  - `'age'` (float): Estimated age.
  - `'face_confidence'` (float): Confidence score for the detected face.
  - `'dominant_gender'` (str): Dominant gender.
  - `'gender'` (dict): Confidence scores for each gender category.
  - `'dominant_emotion'` (str): Dominant emotion.
  - `'emotion'` (dict): Confidence scores for each emotion category.
  - `'dominant_race'` (str): Dominant race.
  - `'race'` (dict): Confidence scores for each race category.

---

## 4. `find`

### Description
Identifies individuals in a database based on a given image.

### Parameters
- **img_path** (Union[str, np.ndarray, IO[bytes]]): The exact path to the image or a numpy array.
- **db_path** (str): Path to the folder containing image files for the database.
- **model_name** (str): Model for face recognition. Default is `"VGG-Face"`.
- **distance_metric** (str): Metric for measuring similarity. Default is `"cosine"`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **align** (bool): Perform alignment based on eye positions. Default is `True`.
- **expand_percentage** (int): Expand detected facial area with a percentage. Default is `0`.
- **threshold** (Optional[float]): Threshold for determining if images represent the same person. Default is `None`.
- **normalization** (str): Normalize the input image. Default is `"base"`.
- **silent** (bool): Suppress log messages. Default is `False`.
- **refresh_database** (bool): Synchronizes the images representation file with the directory. Default is `True`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.
- **batched** (bool): Flag to enable batched processing. Default is `False`.

### Returns
- **Union[List[pd.DataFrame], List[List[Dict[str, Any]]]:** A list of pandas dataframes or a list of dicts containing identity information for detected individuals.

---

## 5. `represent`

### Description
Represents facial images as multi-dimensional vector embeddings.

### Parameters
- **img_path** (Union[str, np.ndarray, IO[bytes], Sequence[Union[str, np.ndarray, IO[bytes]]]): The exact path to the image or a sequence of images.
- **model_name** (str): Model for face recognition. Default is `"VGG-Face"`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **align** (bool): Perform alignment based on eye positions. Default is `True`.
- **expand_percentage** (int): Expand detected facial area with a percentage. Default is `0`.
- **normalization** (str): Normalize the input image. Default is `"base"`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.
- **max_faces** (Optional[int]): Set a limit on the number of faces to be processed. Default is `None`.

### Returns
- **Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]:** A list of dictionaries containing:
  - `'embedding'` (List[float]): Multidimensional vector representing facial features.
  - `'facial_area'` (dict): Detected facial area.
  - `'face_confidence'` (float): Confidence score of face detection.

---

## 6. `stream`

### Description
Runs real-time face recognition and facial attribute analysis.

### Parameters
- **db_path** (str): Path to the folder containing image files for the database.
- **model_name** (str): Model for face recognition. Default is `"VGG-Face"`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **distance_metric** (str): Metric for measuring similarity. Default is `"cosine"`.
- **enable_face_analysis** (bool): Flag to enable face analysis. Default is `True`.
- **source** (Any): The source for the video stream. Default is `0`.
- **time_threshold** (int): Time threshold for face recognition. Default is `5`.
- **frame_threshold** (int): Frame threshold for face recognition. Default is `5`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.
- **output_path** (Optional[str]): Path to save the output video. Default is `None`.
- **debug** (bool): Set to `True` to save frame outcomes. Default is `False`.

### Returns
- **None**: This function does not return a value.

---

## 7. `extract_faces`

### Description
Extracts faces from a given image.

### Parameters
- **img_path** (Union[str, np.ndarray, IO[bytes]]): Path to the image.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **align** (bool): Flag to enable face alignment. Default is `True`.
- **expand_percentage** (int): Expand detected facial area with a percentage. Default is `0`.
- **grayscale** (bool): (Deprecated) Flag to convert the output face image to grayscale. Default is `False`.
- **color_face** (str): Color to return face image output. Default is `"rgb"`.
- **normalize_face** (bool): Flag to enable normalization of the output face image. Default is `True`.
- **anti_spoofing** (bool): Flag to enable anti-spoofing. Default is `False`.

### Returns
- **List[Dict[str, Any]]**: A list of dictionaries, where each dictionary contains:
  - `"face"` (np.ndarray): The detected face as a NumPy array.
  - `"facial_area"` (Dict[str, Any]): The detected face's regions.
  - `"confidence"` (float): The confidence score associated with the detected face.
  - `"is_real"` (boolean): Anti-spoofing analyze result (if enabled).
  - `"antispoof_score"` (float): Score of anti-spoofing analyze result (if enabled).

---

## 8. `cli`

### Description
Command line interface function for the DeepFace library.

### Returns
- **None**: This function does not return a value.

---

## 9. `detectFace` (Deprecated)

### Description
Deprecated face detection function. Use `extract_faces` for the same functionality.

### Parameters
- **img_path** (Union[str, np.ndarray]): Path to the image.
- **target_size** (tuple): Final shape of facial image. Default is `(224, 224)`.
- **detector_backend** (str): Face detector backend. Default is `"opencv"`.
- **enforce_detection** (bool): If no face is detected, raise an exception. Default is `True`.
- **align** (bool): Flag to enable face alignment. Default is `True`.

### Returns
- **Union[np.ndarray, None]**: Detected (and aligned) facial area image as a numpy array or `None` if no face is detected.

### Note
This function is deprecated. It is recommended to use `extract_faces` instead.