# API Documentation for RecognitionPredictor Module

## Classes

### ContinuousBatchInput
A data class that represents the input for a continuous batch of data.

#### Attributes
- `input_ids` (torch.Tensor): Tensor containing the input IDs for the model.
- `attention_mask` (torch.Tensor): Tensor indicating which tokens should be attended to.
- `position_ids` (torch.Tensor): Tensor containing the position IDs for the input tokens.

### ContinuousBatchOutput
A data class that represents the output of a continuous batch processing.

#### Attributes
- `input_ids` (torch.Tensor): Tensor containing the input IDs after processing.
- `preds` (torch.Tensor): Tensor containing the predicted token IDs.
- `bbox_preds` (torch.Tensor): Tensor containing the predicted bounding boxes.
- `done` (torch.Tensor): Tensor indicating which predictions are complete.
- `scores` (torch.Tensor): Tensor containing the confidence scores for the predictions.

### RecognitionPrompt
A data class that represents a prompt for recognition.

#### Attributes
- `id` (int): Unique identifier for the prompt.
- `task_name` (TaskNames): The name of the task associated with the prompt.
- `image` (np.ndarray): The image to be processed.
- `text` (str): The text input associated with the image.
- `math_mode` (bool): Flag indicating whether math mode is enabled.

## Class: RecognitionPredictor

### `__init__(self, checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None)`
Initializes the RecognitionPredictor with the specified parameters.

#### Parameters
- `checkpoint` (optional): Path to the model checkpoint.
- `device` (str): The device to run the model on (default is `settings.TORCH_DEVICE_MODEL`).
- `dtype` (optional): The data type for the model (e.g., `torch.float32`, `torch.float16`).

#### Return Value
None

#### Purpose
Sets up the RecognitionPredictor instance, initializing necessary attributes and loading the model.

### `get_encoder_chunk_size(self) -> int`
Retrieves the encoder chunk size based on the current settings.

#### Return Value
- `int`: The size of the encoder chunk.

#### Purpose
Determines the appropriate chunk size for processing input data based on device settings.

### `setup_cache(self, batch_size: int)`
Sets up the cache for processing a batch of inputs.

#### Parameters
- `batch_size` (int): The size of the batch to be processed.

#### Return Value
None

#### Purpose
Initializes the cache and prepares the prompt queue for the specified batch size.

### `num_empty_slots(self) -> int`
Calculates the number of empty slots in the batch prompt mapping.

#### Return Value
- `int`: The number of empty slots.

#### Purpose
Provides the count of available slots for new prompts in the batch processing.

### `num_active_slots(self) -> int`
Calculates the number of active slots in the batch prompt mapping.

#### Return Value
- `int`: The number of active slots.

#### Purpose
Provides the count of currently active prompts being processed.

### `detect_and_slice_bboxes(self, images: List[Image.Image], task_names: List[str], det_predictor: DetectionPredictor, detection_batch_size: int | None = None, highres_images: List[Image.Image] | None = None) -> dict`
Detects and slices bounding boxes from the provided images.

#### Parameters
- `images` (List[Image.Image]): List of images to process.
- `task_names` (List[str]): List of task names corresponding to each image.
- `det_predictor` (DetectionPredictor): The detection predictor to use for bounding box detection.
- `detection_batch_size` (optional): Batch size for detection (default is None).
- `highres_images` (optional): List of high-resolution images corresponding to the input images.

#### Return Value
- `dict`: A dictionary containing sliced images, slice mapping, polygons, task names, and resolution scales.

#### Purpose
Processes images to detect and slice bounding boxes for further recognition.

### `slice_bboxes(self, images: List[Image.Image], task_names: List[str], bboxes: List[List[List[int]]] | None = None, polygons: List[List[List[List[int]]]] | None = None, input_text: List[List[str | None]] | None = None) -> dict`
Slices bounding boxes or polygons from the provided images.

#### Parameters
- `images` (List[Image.Image]): List of images to process.
- `task_names` (List[str]): List of task names corresponding to each image.
- `bboxes` (optional): List of bounding boxes for each image.
- `polygons` (optional): List of polygons for each image.
- `input_text` (optional): List of input text for each image.

#### Return Value
- `dict`: A dictionary containing sliced images, slice mapping, polygons, input text, task names, and resolution scales.

#### Purpose
Extracts slices from images based on provided bounding boxes or polygons.

### `prepare_input(self, task_names: List[str], images: List[Image.Image], input_text: List[str | None], math_modes: List[bool])`
Prepares the input for the model based on the provided parameters.

#### Parameters
- `task_names` (List[str]): List of task names for each image.
- `images` (List[Image.Image]): List of images to process.
- `input_text` (List[str | None]): List of input text for each image.
- `math_modes` (List[bool]): List of flags indicating math mode for each image.

#### Return Value
- `List[dict]`: A list of dictionaries containing the prepared input for each task.

#### Purpose
Formats the input data for the model based on the specified tasks and images.

### `process_outputs(self, outputs: SuryaModelOutput) -> ContinuousBatchOutput`
Processes the model outputs to extract relevant information.

#### Parameters
- `outputs` (SuryaModelOutput): The raw outputs from the model.

#### Return Value
- `ContinuousBatchOutput`: An object containing processed input IDs, predictions, bounding box predictions, completion status, and scores.

#### Purpose
Transforms the raw model outputs into a structured format for further processing.

### `decode(self, current_inputs: Optional[ContinuousBatchInput] = None)`
Decodes the current inputs to generate predictions.

#### Parameters
- `current_inputs` (optional): The current batch input to decode.

#### Return Value
- `Tuple[ContinuousBatchInput, ContinuousBatchOutput]`: A tuple containing the new input and the processed output.

#### Purpose
Generates predictions from the model based on the current inputs.

### `prefill(self, current_inputs: Optional[ContinuousBatchInput] = None)`
Prefills the empty slots in the batch with new inputs.

#### Parameters
- `current_inputs` (optional): The current batch input to prefill.

#### Return Value
- `Tuple[ContinuousBatchInput, ContinuousBatchOutput, List[int]]`: A tuple containing the new input, processed outputs, and indices of merged prompts.

#### Purpose
Fills empty slots in the batch with new prompts and processes them.

### `maybe_trim_cache_padding(self, current_inputs: ContinuousBatchInput) -> ContinuousBatchInput`
Trims the padding from the cache and attention mask if possible.

#### Parameters
- `current_inputs` (ContinuousBatchInput): The current batch input to potentially trim.

#### Return Value
- `ContinuousBatchInput`: The trimmed batch input.

#### Purpose
Reduces unnecessary padding in the cache to optimize processing.

### `prediction_loop(self, flat: dict, recognition_batch_size: int | None = None, math_mode: bool = True) -> tuple`
Runs the prediction loop for the recognition process.

#### Parameters
- `flat` (dict): A dictionary containing the flattened input data.
- `recognition_batch_size` (optional): The batch size for recognition (default is None).
- `math_mode` (bool): Flag indicating whether math mode is enabled.

#### Return Value
- `Tuple[list, torch.Tensor, list]`: A tuple containing predicted tokens, bounding boxes, and scores.

#### Purpose
Executes the recognition process in a loop until all prompts are processed.

### `get_bboxes_text(self, flat: dict, predicted_tokens: list, scores: list, predicted_polygons: list, drop_repeated_text: bool = False) -> list`
Extracts text and bounding boxes from the predicted tokens.

#### Parameters
- `flat` (dict): A dictionary containing the flattened input data.
- `predicted_tokens` (list): List of predicted token IDs.
- `scores` (list): List of confidence scores for the predictions.
- `predicted_polygons` (list): List of predicted polygons for the bounding boxes.
- `drop_repeated_text` (bool): Flag indicating whether to drop repeated text.

#### Return Value
- `list`: A list of character predictions extracted from the predicted tokens.

#### Purpose
Processes the predicted tokens to generate structured text and bounding box information.

### `__call__(self, images: List[Image.Image], task_names: List[str] | None = None, det_predictor: DetectionPredictor | None = None, detection_batch_size: int | None = None, recognition_batch_size: int | None = None, highres_images: List[Image.Image] | None = None, bboxes: List[List[List[int]]] | None = None, polygons: List[List[List[List[int]]]] | None = None, input_text: List[List[str | None]] | None = None, sort_lines: bool = False, math_mode: bool = True, return_words: bool = False, drop_repeated_text: bool = False) -> List[OCRResult]`
Processes the provided images to extract text and bounding boxes.

#### Parameters
- `images` (List[Image.Image]): List of images to process.
- `task_names` (optional): List of task names corresponding to each image.
- `det_predictor` (optional): The detection predictor to use for bounding box detection.
- `detection_batch_size` (optional): Batch size for detection.
- `recognition_batch_size` (optional): Batch size for recognition.
- `highres_images` (optional): List of high-resolution images.
- `bboxes` (optional): List of bounding boxes for each image.
- `polygons` (optional): List of polygons for each image.
- `input_text` (optional): List of input text for each image.
- `sort_lines` (bool): Flag indicating whether to sort lines.
- `math_mode` (bool): Flag indicating whether math mode is enabled.
- `return_words` (bool): Flag indicating whether to return words.
- `drop_repeated_text` (bool): Flag indicating whether to drop repeated text.

#### Return Value
- `List[OCRResult]`: A list of OCR results containing extracted text and bounding boxes.

#### Purpose
Main entry point for processing images and extracting text and bounding boxes based on the specified tasks and parameters.

