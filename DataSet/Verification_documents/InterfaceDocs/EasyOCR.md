# API Documentation for EasyOCR Reader Class

## Class: `Reader`

The `Reader` class provides an interface for Optical Character Recognition (OCR) using the EasyOCR library. It supports multiple languages and utilizes deep learning models for text detection and recognition.

### Attributes:
- `verbose` (bool): If True, enables verbose logging.
- `download_enabled` (bool): If True, allows downloading of model data.
- `model_storage_directory` (str): Directory path for storing model data.
- `user_network_directory` (str): Directory path for custom network architecture.
- `device` (str): Device type ('cpu', 'cuda', or 'mps') for model execution.
- `detection_models` (dict): Dictionary of available detection models.
- `recognition_models` (dict): Dictionary of available recognition models.
- `character` (str): Characters supported by the recognition model.
- `detector`: The initialized text detector.
- `recognizer`: The initialized text recognizer.
- `converter`: The converter for the recognition model.

### Method: `__init__(self, lang_list, gpu=True, model_storage_directory=None, user_network_directory=None, detect_network="craft", recog_network='standard', download_enabled=True, detector=True, recognizer=True, verbose=True, quantize=True, cudnn_benchmark=False)`

#### Parameters:
- `lang_list` (list): List of language codes (ISO 639) for languages to be recognized.
- `gpu` (bool): Enable GPU support (default is True).
- `model_storage_directory` (str, optional): Path to directory for model data.
- `user_network_directory` (str, optional): Path to directory for custom network architecture.
- `detect_network` (str): Detection network to use ('craft' or 'dbnet18').
- `recog_network` (str): Recognition network to use ('standard' or other supported models).
- `download_enabled` (bool): Enable downloading of model data via HTTP (default is True).
- `detector` (bool): If True, initializes the detector.
- `recognizer` (bool): If True, initializes the recognizer.
- `verbose` (bool): If True, enables verbose logging.
- `quantize` (bool): If True, enables model quantization.
- `cudnn_benchmark` (bool): If True, enables cuDNN benchmark mode.

#### Return Value:
- None

#### Description:
Initializes the `Reader` object, setting up the necessary configurations for language recognition, model paths, and device settings.

---

### Method: `getDetectorPath(self, detect_network)`

#### Parameters:
- `detect_network` (str): The name of the detection network to retrieve the path for.

#### Return Value:
- (str): The path to the detection model.

#### Description:
Retrieves the path for the specified detection network and checks if the model file exists. If not, it downloads the model if downloading is enabled.

---

### Method: `initDetector(self, detector_path)`

#### Parameters:
- `detector_path` (str): The path to the detection model.

#### Return Value:
- The initialized detector object.

#### Description:
Initializes the text detector using the specified model path.

---

### Method: `setDetector(self, detect_network)`

#### Parameters:
- `detect_network` (str): The name of the detection network to set.

#### Return Value:
- None

#### Description:
Sets the detection network and initializes the detector.

---

### Method: `setModelLanguage(self, language, lang_list, list_lang, list_lang_string)`

#### Parameters:
- `language` (str): The language to set for the model.
- `lang_list` (list): List of language codes.
- `list_lang` (list): List of supported languages for the specified model.
- `list_lang_string` (str): String representation of the supported languages.

#### Return Value:
- None

#### Description:
Sets the language for the recognition model and checks compatibility with the specified language list.

---

### Method: `getChar(self, fileName)`

#### Parameters:
- `fileName` (str): The name of the character file to read.

#### Return Value:
- (str): A string of characters read from the file.

#### Description:
Reads a character file and returns the characters as a string.

---

### Method: `setLanguageList(self, lang_list, model)`

#### Parameters:
- `lang_list` (list): List of language codes.
- `model` (dict): The model configuration dictionary.

#### Return Value:
- None

#### Description:
Sets the list of characters supported by the specified languages.

---

### Method: `detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, add_margin=0.1, reformat=True, optimal_num_chars=None, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0)`

#### Parameters:
- `img` (numpy.ndarray): The input image for text detection.
- `min_size` (int): Minimum size of detected text boxes (default is 20).
- `text_threshold` (float): Threshold for text confidence (default is 0.7).
- `low_text` (float): Threshold for low text confidence (default is 0.4).
- `link_threshold` (float): Threshold for linking text boxes (default is 0.4).
- `canvas_size` (int): Size of the canvas for detection (default is 2560).
- `mag_ratio` (float): Magnification ratio for the image (default is 1.0).
- `slope_ths` (float): Slope threshold for grouping text boxes (default is 0.1).
- `ycenter_ths` (float): Y-center threshold for grouping (default is 0.5).
- `height_ths` (float): Height threshold for grouping (default is 0.5).
- `width_ths` (float): Width threshold for grouping (default is 0.5).
- `add_margin` (float): Margin to add around detected boxes (default is 0.1).
- `reformat` (bool): If True, reformats the input image (default is True).
- `optimal_num_chars` (int, optional): Optimal number of characters for detection.
- `threshold` (float): Threshold for detection (default is 0.2).
- `bbox_min_score` (float): Minimum score for bounding boxes (default is 0.2).
- `bbox_min_size` (int): Minimum size for bounding boxes (default is 3).
- `max_candidates` (int): Maximum number of candidates for detection (default is 0).

#### Return Value:
- (list, list): A tuple containing two lists of detected text boxes.

#### Description:
Detects text in the input image and returns the bounding boxes of detected text regions.

---

### Method: `recognize(self, img_cv_grey, horizontal_list=None, free_list=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, y_ths=0.5, x_ths=1.0, reformat=True, output_format='standard')`

#### Parameters:
- `img_cv_grey` (numpy.ndarray): The input image in grayscale for text recognition.
- `horizontal_list` (list, optional): List of horizontal bounding boxes.
- `free_list` (list, optional): List of free-form bounding boxes.
- `decoder` (str): The decoding method to use ('greedy' or 'beam').
- `beamWidth` (int): Width for beam search (default is 5).
- `batch_size` (int): Number of images to process in a batch (default is 1).
- `workers` (int): Number of worker threads for processing (default is 0).
- `allowlist` (str, optional): Characters to allow in recognition.
- `blocklist` (str, optional): Characters to block in recognition.
- `detail` (int): Level of detail in the output (default is 1).
- `rotation_info` (list, optional): Information about image rotations.
- `paragraph` (bool): If True, returns results in paragraph format.
- `contrast_ths` (float): Threshold for contrast adjustment (default is 0.1).
- `adjust_contrast` (float): Amount to adjust contrast (default is 0.5).
- `filter_ths` (float): Threshold for filtering results (default is 0.003).
- `y_ths` (float): Y-threshold for paragraph grouping (default is 0.5).
- `x_ths` (float): X-threshold for paragraph grouping (default is 1.0).
- `reformat` (bool): If True, reformats the input image (default is True).
- `output_format` (str): Format of the output ('standard', 'dict', 'json', 'free_merge').

#### Return Value:
- (list): A list of recognized text results.

#### Description:
Recognizes text in the input image based on the provided bounding boxes and returns the recognized text along with its confidence scores.

---

### Method: `readtext(self, image, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

#### Parameters:
- `image` (str or numpy.ndarray): File path or image array for text recognition.
- `decoder` (str): The decoding method to use (default is 'greedy').
- `beamWidth` (int): Width for beam search (default is 5).
- `batch_size` (int): Number of images to process in a batch (default is 1).
- `workers` (int): Number of worker threads for processing (default is 0).
- `allowlist` (str, optional): Characters to allow in recognition.
- `blocklist` (str, optional): Characters to block in recognition.
- `detail` (int): Level of detail in the output (default is 1).
- `rotation_info` (list, optional): Information about image rotations.
- `paragraph` (bool): If True, returns results in paragraph format.
- `min_size` (int): Minimum size of detected text boxes (default is 20).
- `contrast_ths` (float): Threshold for contrast adjustment (default is 0.1).
- `adjust_contrast` (float): Amount to adjust contrast (default is 0.5).
- `filter_ths` (float): Threshold for filtering results (default is 0.003).
- `text_threshold` (float): Threshold for text confidence (default is 0.7).
- `low_text` (float): Threshold for low text confidence (default is 0.4).
- `link_threshold` (float): Threshold for linking text boxes (default is 0.4).
- `canvas_size` (int): Size of the canvas for detection (default is 2560).
- `mag_ratio` (float): Magnification ratio for the image (default is 1.0).
- `slope_ths` (float): Slope threshold for grouping text boxes (default is 0.1).
- `ycenter_ths` (float): Y-center threshold for grouping (default is 0.5).
- `height_ths` (float): Height threshold for grouping (default is 0.5).
- `width_ths` (float): Width threshold for grouping (default is 0.5).
- `y_ths` (float): Y-threshold for paragraph grouping (default is 0.5).
- `x_ths` (float): X-threshold for paragraph grouping (default is 1.0).
- `add_margin` (float): Margin to add around detected boxes (default is 0.1).
- `threshold` (float): Threshold for detection (default is 0.2).
- `bbox_min_score` (float): Minimum score for bounding boxes (default is 0.2).
- `bbox_min_size` (int): Minimum size for bounding boxes (default is 3).
- `max_candidates` (int): Maximum number of candidates for detection (default is 0).
- `output_format` (str): Format of the output (default is 'standard').

#### Return Value:
- (list): A list of recognized text results.

#### Description:
Reads text from the provided image, detects text regions, and recognizes the text within those regions.

---

### Method: `readtextlang(self, image, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

#### Parameters:
- `image` (str or numpy.ndarray): File path or image array for text recognition.
- `decoder` (str): The decoding method to use (default is 'greedy').
- `beamWidth` (int): Width for beam search (default is 5).
- `batch_size` (int): Number of images to process in a batch (default is 1).
- `workers` (int): Number of worker threads for processing (default is 0).
- `allowlist` (str, optional): Characters to allow in recognition.
- `blocklist` (str, optional): Characters to block in recognition.
- `detail` (int): Level of detail in the output (default is 1).
- `rotation_info` (list, optional): Information about image rotations.
- `paragraph` (bool): If True, returns results in paragraph format.
- `min_size` (int): Minimum size of detected text boxes (default is 20).
- `contrast_ths` (float): Threshold for contrast adjustment (default is 0.1).
- `adjust_contrast` (float): Amount to adjust contrast (default is 0.5).
- `filter_ths` (float): Threshold for filtering results (default is 0.003).
- `text_threshold` (float): Threshold for text confidence (default is 0.7).
- `low_text` (float): Threshold for low text confidence (default is 0.4).
- `link_threshold` (float): Threshold for linking text boxes (default is 0.4).
- `canvas_size` (int): Size of the canvas for detection (default is 2560).
- `mag_ratio` (float): Magnification ratio for the image (default is 1.0).
- `slope_ths` (float): Slope threshold for grouping text boxes (default is 0.1).
- `ycenter_ths` (float): Y-center threshold for grouping (default is 0.5).
- `height_ths` (float): Height threshold for grouping (default is 0.5).
- `width_ths` (float): Width threshold for grouping (default is 0.5).
- `y_ths` (float): Y-threshold for paragraph grouping (default is 0.5).
- `x_ths` (float): X-threshold for paragraph grouping (default is 1.0).
- `add_margin` (float): Margin to add around detected boxes (default is 0.1).
- `threshold` (float): Threshold for detection (default is 0.2).
- `bbox_min_score` (float): Minimum score for bounding boxes (default is 0.2).
- `bbox_min_size` (int): Minimum size for bounding boxes (default is 3).
- `max_candidates` (int): Maximum number of candidates for detection (default is 0).
- `output_format` (str): Format of the output (default is 'standard').

#### Return Value:
- (list): A list of recognized text results.

#### Description:
Reads text from the provided image, detects text regions, recognizes the text within those regions, and associates the recognized text with its corresponding language.

---

### Method: `readtext_batched(self, image, n_width=None, n_height=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

#### Parameters:
- `image` (list): List of file paths or image arrays for text recognition.
- `n_width` (int, optional): New width for resizing images.
- `n_height` (int, optional): New height for resizing images.
- `decoder` (str): The decoding method to use (default is 'greedy').
- `beamWidth` (int): Width for beam search (default is 5).
- `batch_size` (int): Number of images to process in a batch (default is 1).
- `workers` (int): Number of worker threads for processing (default is 0).
- `allowlist` (str, optional): Characters to allow in recognition.
- `blocklist` (str, optional): Characters to block in recognition.
- `detail` (int): Level of detail in the output (default is 1).
- `rotation_info` (list, optional): Information about image rotations.
- `paragraph` (bool): If True, returns results in paragraph format.
- `min_size` (int): Minimum size of detected text boxes (default is 20).
- `contrast_ths` (float): Threshold for contrast adjustment (default is 0.1).
- `adjust_contrast` (float): Amount to adjust contrast (default is 0.5).
- `filter_ths` (float): Threshold for filtering results (default is 0.003).
- `text_threshold` (float): Threshold for text confidence (default is 0.7).
- `low_text` (float): Threshold for low text confidence (default is 0.4).
- `link_threshold` (float): Threshold for linking text boxes (default is 0.4).
- `canvas_size` (int): Size of the canvas for detection (default is 2560).
- `mag_ratio` (float): Magnification ratio for the image (default is 1.0).
- `slope_ths` (float): Slope threshold for grouping text boxes (default is 0.1).
- `ycenter_ths` (float): Y-center threshold for grouping (default is 0.5).
- `height_ths` (float): Height threshold for grouping (default is 0.5).
- `width_ths` (float): Width threshold for grouping (default is 0.5).
- `y_ths` (float): Y-threshold for paragraph grouping (default is 0.5).
- `x_ths` (float): X-threshold for paragraph grouping (default is 1.0).
- `add_margin` (float): Margin to add around detected boxes (default is 0.1).
- `threshold` (float): Threshold for detection (default is 0.2).
- `bbox_min_score` (float): Minimum score for bounding boxes (default is 0.2).
- `bbox_min_size` (int): Minimum size for bounding boxes (default is 3).
- `max_candidates` (int): Maximum number of candidates for detection (default is 0).
- `output_format` (str): Format of the output (default is 'standard').

#### Return Value:
- (list): A list of recognized text results for each image.

#### Description:
Processes a batch of images for text detection and recognition, returning the results for each image in the specified format.