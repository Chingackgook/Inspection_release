# EasyOCR API Documentation

## Class: `Reader`

### Initialization

#### `__init__(self, lang_list, gpu=True, model_storage_directory=None, user_network_directory=None, detect_network="craft", recog_network='standard', download_enabled=True, detector=True, recognizer=True, verbose=True, quantize=True, cudnn_benchmark=False)`

**Parameters:**
- `lang_list`: List of languages to be used for text recognition (e.g., `['en', 'ch_sim']`).
- `gpu`: Boolean indicating whether to use GPU (default is `True`).
- `model_storage_directory`: Directory to store models (default is `None`).
- `user_network_directory`: Directory for user-defined networks (default is `None`).
- `detect_network`: Detection network to use (default is `"craft"`).
- `recog_network`: Recognition network to use (default is `'standard'`).
- `download_enabled`: Boolean indicating if model downloads are enabled (default is `True`).
- `detector`: Boolean indicating if the detector should be initialized (default is `True`).
- `recognizer`: Boolean indicating if the recognizer should be initialized (default is `True`).
- `verbose`: Boolean indicating if verbose logging is enabled (default is `True`).
- `quantize`: Boolean indicating if model quantization is enabled (default is `True`).
- `cudnn_benchmark`: Boolean indicating if cuDNN benchmark is enabled (default is `False`).

**Returns:** None

---

### Method: `getDetectorPath(self, detect_network)`

**Parameters:**
- `detect_network`: The name of the detection network to retrieve the path for.

**Returns:**
- `detector_path`: The file path of the detection model.

---

### Method: `initDetector(self, detector_path)`

**Parameters:**
- `detector_path`: The file path of the detection model.

**Returns:**
- `detector`: Initialized detector object.

---

### Method: `setDetector(self, detect_network)`

**Parameters:**
- `detect_network`: The name of the detection network to set.

**Returns:** None

---

### Method: `setModelLanguage(self, language, lang_list, list_lang, list_lang_string)`

**Parameters:**
- `language`: The language to set for the model.
- `lang_list`: List of languages being used.
- `list_lang`: List of supported languages.
- `list_lang_string`: String representation of supported languages.

**Returns:** None

---

### Method: `getChar(self, fileName)`

**Parameters:**
- `fileName`: The name of the character file to read.

**Returns:**
- `char`: A string containing characters from the file.

---

### Method: `setLanguageList(self, lang_list, model)`

**Parameters:**
- `lang_list`: List of languages being used.
- `model`: The model containing character information.

**Returns:** None

---

### Method: `detect(self, img, min_size=20, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, add_margin=0.1, reformat=True, optimal_num_chars=None, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0)`

**Parameters:**
- `img`: Input image for text detection.
- `min_size`: Minimum size of detected text (default is `20`).
- `text_threshold`: Threshold for text detection (default is `0.7`).
- `low_text`: Low text threshold (default is `0.4`).
- `link_threshold`: Link threshold (default is `0.4`).
- `canvas_size`: Size of the canvas (default is `2560`).
- `mag_ratio`: Magnification ratio (default is `1.`).
- `slope_ths`: Slope threshold (default is `0.1`).
- `ycenter_ths`: Y-center threshold (default is `0.5`).
- `height_ths`: Height threshold (default is `0.5`).
- `width_ths`: Width threshold (default is `0.5`).
- `add_margin`: Margin to add (default is `0.1`).
- `reformat`: Boolean indicating if the image should be reformatted (default is `True`).
- `optimal_num_chars`: Optimal number of characters (default is `None`).
- `threshold`: Threshold for bounding box (default is `0.2`).
- `bbox_min_score`: Minimum score for bounding box (default is `0.2`).
- `bbox_min_size`: Minimum size for bounding box (default is `3`).
- `max_candidates`: Maximum number of candidates (default is `0`).

**Returns:**
- `horizontal_list_agg`: Aggregated list of horizontal text boxes.
- `free_list_agg`: Aggregated list of free text boxes.

---

### Method: `recognize(self, img_cv_grey, horizontal_list=None, free_list=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, y_ths=0.5, x_ths=1.0, reformat=True, output_format='standard')`

**Parameters:**
- `img_cv_grey`: Grayscale image for recognition.
- `horizontal_list`: List of horizontal bounding boxes (default is `None`).
- `free_list`: List of free bounding boxes (default is `None`).
- `decoder`: Decoding method (default is `'greedy'`).
- `beamWidth`: Width of the beam for decoding (default is `5`).
- `batch_size`: Size of the batch for processing (default is `1`).
- `workers`: Number of workers for parallel processing (default is `0`).
- `allowlist`: List of allowed characters (default is `None`).
- `blocklist`: List of blocked characters (default is `None`).
- `detail`: Level of detail in the output (default is `1`).
- `rotation_info`: Information about image rotation (default is `None`).
- `paragraph`: Boolean indicating if paragraph detection is enabled (default is `False`).
- `contrast_ths`: Threshold for contrast adjustment (default is `0.1`).
- `adjust_contrast`: Amount to adjust contrast (default is `0.5`).
- `filter_ths`: Threshold for filtering (default is `0.003`).
- `y_ths`: Y-threshold for paragraph detection (default is `0.5`).
- `x_ths`: X-threshold for paragraph detection (default is `1.0`).
- `reformat`: Boolean indicating if the image should be reformatted (default is `True`).
- `output_format`: Format of the output (default is `'standard'`).

**Returns:**
- `result`: List of recognized text and their bounding boxes.

---

### Method: `readtext(self, image, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

**Parameters:**
- `image`: File path, numpy array, or byte stream object of the image.
- `decoder`: Decoding method (default is `'greedy'`).
- `beamWidth`: Width of the beam for decoding (default is `5`).
- `batch_size`: Size of the batch for processing (default is `1`).
- `workers`: Number of workers for parallel processing (default is `0`).
- `allowlist`: List of allowed characters (default is `None`).
- `blocklist`: List of blocked characters (default is `None`).
- `detail`: Level of detail in the output (default is `1`).
- `rotation_info`: Information about image rotation (default is `None`).
- `paragraph`: Boolean indicating if paragraph detection is enabled (default is `False`).
- `min_size`: Minimum size of detected text (default is `20`).
- `contrast_ths`: Threshold for contrast adjustment (default is `0.1`).
- `adjust_contrast`: Amount to adjust contrast (default is `0.5`).
- `filter_ths`: Threshold for filtering (default is `0.003`).
- `text_threshold`: Threshold for text detection (default is `0.7`).
- `low_text`: Low text threshold (default is `0.4`).
- `link_threshold`: Link threshold (default is `0.4`).
- `canvas_size`: Size of the canvas (default is `2560`).
- `mag_ratio`: Magnification ratio (default is `1.`).
- `slope_ths`: Slope threshold (default is `0.1`).
- `ycenter_ths`: Y-center threshold (default is `0.5`).
- `height_ths`: Height threshold (default is `0.5`).
- `width_ths`: Width threshold (default is `0.5`).
- `y_ths`: Y-threshold for paragraph detection (default is `0.5`).
- `x_ths`: X-threshold for paragraph detection (default is `1.0`).
- `add_margin`: Margin to add (default is `0.1`).
- `threshold`: Threshold for bounding box (default is `0.2`).
- `bbox_min_score`: Minimum score for bounding box (default is `0.2`).
- `bbox_min_size`: Minimum size for bounding box (default is `3`).
- `max_candidates`: Maximum number of candidates (default is `0`).
- `output_format`: Format of the output (default is `'standard'`).

**Returns:**
- `result`: List of recognized text and their bounding boxes.

---

### Method: `readtextlang(self, image, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

**Parameters:**
- `image`: File path, numpy array, or byte stream object of the image.
- `decoder`: Decoding method (default is `'greedy'`).
- `beamWidth`: Width of the beam for decoding (default is `5`).
- `batch_size`: Size of the batch for processing (default is `1`).
- `workers`: Number of workers for parallel processing (default is `0`).
- `allowlist`: List of allowed characters (default is `None`).
- `blocklist`: List of blocked characters (default is `None`).
- `detail`: Level of detail in the output (default is `1`).
- `rotation_info`: Information about image rotation (default is `None`).
- `paragraph`: Boolean indicating if paragraph detection is enabled (default is `False`).
- `min_size`: Minimum size of detected text (default is `20`).
- `contrast_ths`: Threshold for contrast adjustment (default is `0.1`).
- `adjust_contrast`: Amount to adjust contrast (default is `0.5`).
- `filter_ths`: Threshold for filtering (default is `0.003`).
- `text_threshold`: Threshold for text detection (default is `0.7`).
- `low_text`: Low text threshold (default is `0.4`).
- `link_threshold`: Link threshold (default is `0.4`).
- `canvas_size`: Size of the canvas (default is `2560`).
- `mag_ratio`: Magnification ratio (default is `1.`).
- `slope_ths`: Slope threshold (default is `0.1`).
- `ycenter_ths`: Y-center threshold (default is `0.5`).
- `height_ths`: Height threshold (default is `0.5`).
- `width_ths`: Width threshold (default is `0.5`).
- `y_ths`: Y-threshold for paragraph detection (default is `0.5`).
- `x_ths`: X-threshold for paragraph detection (default is `1.0`).
- `add_margin`: Margin to add (default is `0.1`).
- `threshold`: Threshold for bounding box (default is `0.2`).
- `bbox_min_score`: Minimum score for bounding box (default is `0.2`).
- `bbox_min_size`: Minimum size for bounding box (default is `3`).
- `max_candidates`: Maximum number of candidates (default is `0`).
- `output_format`: Format of the output (default is `'standard'`).

**Returns:**
- `result`: List of recognized text and their bounding boxes.

---

### Method: `readtext_batched(self, image, n_width=None, n_height=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1, rotation_info=None, paragraph=False, min_size=20, contrast_ths=0.1, adjust_contrast=0.5, filter_ths=0.003, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560, mag_ratio=1., slope_ths=0.1, ycenter_ths=0.5, height_ths=0.5, width_ths=0.5, y_ths=0.5, x_ths=1.0, add_margin=0.1, threshold=0.2, bbox_min_score=0.2, bbox_min_size=3, max_candidates=0, output_format='standard')`

**Parameters:**
- `image`: File path, numpy array, or byte stream object of the image or a list of images.
- `n_width`: New width for resizing (default is `None`).
- `n_height`: New height for resizing (default is `None`).
- `decoder`: Decoding method (default is `'greedy'`).
- `beamWidth`: Width of the beam for decoding (default is `5`).
- `batch_size`: Size of the batch for processing (default is `1`).
- `workers`: Number of workers for parallel processing (default is `0`).
- `allowlist`: List of allowed characters (default is `None`).
- `blocklist`: List of blocked characters (default is `None`).
- `detail`: Level of detail in the output (default is `1`).
- `rotation_info`: Information about image rotation (default is `None`).
- `paragraph`: Boolean indicating if paragraph detection is enabled (default is `False`).
- `min_size`: Minimum size of detected text (default is `20`).
- `contrast_ths`: Threshold for contrast adjustment (default is `0.1`).
- `adjust_contrast`: Amount to adjust contrast (default is `0.5`).
- `filter_ths`: Threshold for filtering (default is `0.003`).
- `text_threshold`: Threshold for text detection (default is `0.7`).
- `low_text`: Low text threshold (default is `0.4`).
- `link_threshold`: Link threshold (default is `0.4`).
- `canvas_size`: Size of the canvas (default is `2560`).
- `mag_ratio`: Magnification ratio (default is `1.`).
- `slope_ths`: Slope threshold (default is `0.1`).
- `ycenter_ths`: Y-center threshold (default is `0.5`).
- `height_ths`: Height threshold (default is `0.5`).
- `width_ths`: Width threshold (default is `0.5`).
- `y_ths`: Y-threshold for paragraph detection (default is `0.5`).
- `x_ths`: X-threshold for paragraph detection (default is `1.0`).
- `add_margin`: Margin to add (default is `0.1`).
- `threshold`: Threshold for bounding box (default is `0.2`).
- `bbox_min_score`: Minimum score for bounding box (default is `0.2`).
- `bbox_min_size`: Minimum size for bounding box (default is `3`).
- `max_candidates`: Maximum number of candidates (default is `0`).
- `output_format`: Format of the output (default is `'standard'`).

**Returns:**
- `result_agg`: Aggregated results of recognized text and their bounding boxes for all images.

---

## Example Usage

```python
import easyocr

# Initialize the Reader
reader = easyocr.Reader(['ch_sim', 'en'])  # This needs to run only once to load the model into memory

# Read text from an image
result = reader.readtext('/path/to/image.jpg')

# Print the result
print(result)
```