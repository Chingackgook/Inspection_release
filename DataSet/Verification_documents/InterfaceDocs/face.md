# API Documentation

## Function: `face_distance`

### Description
Calculates the Euclidean distance between a list of face encodings and a single face encoding to determine how similar the faces are.

### Parameters
- **face_encodings** (List[np.ndarray]): A list of face encodings to compare.
- **face_to_compare** (np.ndarray): A single face encoding to compare against.

### Returns
- **np.ndarray**: A numpy array containing the distance for each face in the same order as the `face_encodings` array.

### Parameter Value Range
- `face_encodings`: List can be empty or contain multiple face encodings.
- `face_to_compare`: Should be a valid face encoding (1D numpy array of shape (128,)).

### Purpose
To determine the similarity between faces based on their encodings, where a smaller distance indicates a higher similarity.

---

## Function: `load_image_file`

### Description
Loads an image file into a numpy array.

### Parameters
- **file** (str or file object): The image file name or file object to load.
- **mode** (str): The format to convert the image to. Supported values are 'RGB' (default) and 'L' (black and white).

### Returns
- **np.ndarray**: The image contents as a numpy array.

### Parameter Value Range
- `mode`: Must be either 'RGB' or 'L'.

### Purpose
To read an image file and convert it into a format suitable for face recognition processing.

---

## Function: `face_locations`

### Description
Returns the bounding boxes of human faces in an image.

### Parameters
- **img** (np.ndarray): An image represented as a numpy array.
- **number_of_times_to_upsample** (int): How many times to upsample the image looking for faces. Higher numbers find smaller faces. Default is 1.
- **model** (str): The face detection model to use. Options are "hog" (default) or "cnn".

### Returns
- **List[Tuple[int, int, int, int]]**: A list of tuples representing the found face locations in (top, right, bottom, left) order.

### Parameter Value Range
- `number_of_times_to_upsample`: Must be a non-negative integer.
- `model`: Must be either "hog" or "cnn".

### Purpose
To detect and return the locations of faces within an image.

---

## Function: `batch_face_locations`

### Description
Returns the bounding boxes of human faces in a batch of images using the CNN face detector.

### Parameters
- **images** (List[np.ndarray]): A list of images, each represented as a numpy array.
- **number_of_times_to_upsample** (int): How many times to upsample the images looking for faces. Default is 1.
- **batch_size** (int): How many images to include in each GPU processing batch. Default is 128.

### Returns
- **List[List[Tuple[int, int, int, int]]]**: A list of lists, where each inner list contains tuples representing the found face locations in (top, right, bottom, left) order for each image.

### Parameter Value Range
- `number_of_times_to_upsample`: Must be a non-negative integer.
- `batch_size`: Must be a positive integer.

### Purpose
To efficiently detect faces in multiple images at once, leveraging GPU processing for speed.

---

## Function: `face_landmarks`

### Description
Returns a dictionary of face feature locations (e.g., eyes, nose) for each detected face in an image.

### Parameters
- **face_image** (np.ndarray): The image to search for faces.
- **face_locations** (Optional[List[Tuple[int, int, int, int]]]): A list of face locations to check. If not provided, all faces in the image will be processed.
- **model** (str): The model to use for landmark detection. Options are "large" (default) or "small".

### Returns
- **List[Dict[str, List[Tuple[int, int]]]]**: A list of dictionaries containing the locations of facial features for each detected face.

### Parameter Value Range
- `model`: Must be either "large" or "small".

### Purpose
To identify and return the key facial features for each detected face in an image.

---

## Function: `face_encodings`

### Description
Returns the 128-dimensional face encoding for each face detected in an image.

### Parameters
- **face_image** (np.ndarray): The image containing one or more faces.
- **known_face_locations** (Optional[List[Tuple[int, int, int, int]]]): The bounding boxes of each face if already known.
- **num_jitters** (int): How many times to re-sample the face when calculating encoding. Higher values yield more accurate results but are slower. Default is 1.
- **model** (str): The model to use for encoding. Options are "large" or "small" (default).

### Returns
- **List[np.ndarray]**: A list of 128-dimensional face encodings, one for each detected face.

### Parameter Value Range
- `num_jitters`: Must be a non-negative integer.
- `model`: Must be either "large" or "small".

### Purpose
To generate a numerical representation of each face that can be used for comparison and recognition.

---

## Function: `compare_faces`

### Description
Compares a list of known face encodings against a candidate encoding to determine if they match.

### Parameters
- **known_face_encodings** (List[np.ndarray]): A list of known face encodings.
- **face_encoding_to_check** (np.ndarray): A single face encoding to compare against the list.
- **tolerance** (float): The distance threshold for considering a match. Lower values are more strict. Default is 0.6.

### Returns
- **List[bool]**: A list of boolean values indicating which known face encodings match the face encoding to check.

### Parameter Value Range
- `tolerance`: Must be a non-negative float.

### Purpose
To determine if a given face encoding matches any of the known face encodings based on a specified tolerance level.