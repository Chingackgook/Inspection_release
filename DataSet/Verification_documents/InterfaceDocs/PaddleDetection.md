# API Documentation

## Class: Trainer
The `Trainer` class is a core component in the PaddlePaddle framework for training, evaluating, and inferring deep learning models. It provides methods for model training, evaluation, and prediction.

### `__init__(self, cfg, mode='train')`
#### Parameters:
- `cfg` (dict): Configuration dictionary containing settings for the trainer, model, dataset, and other parameters.
- `mode` (str): Mode of operation. It can be one of the following:
  - `'train'`: For training the model.
  - `'eval'`: For evaluating the model.
  - `'test'`: For testing the model.

#### Attributes:
- `self.cfg`: A copy of the configuration dictionary.
- `self.mode`: The mode of operation (train, eval, test).
- `self.optimizer`: The optimizer used for training.
- `self.is_loaded_weights`: A flag indicating if weights have been loaded.
- `self.use_amp`: A flag indicating if automatic mixed precision is used.
- `self.amp_level`: The level of automatic mixed precision.
- `self.custom_white_list`: Custom white list for mixed precision.
- `self.custom_black_list`: Custom black list for mixed precision.
- `self.use_master_grad`: A flag indicating if master gradient is used.
- `self.uniform_output_enabled`: A flag indicating if uniform output is enabled.
- `self.dataset`: The dataset used for training, evaluation, or testing.
- `self.loader`: The data loader for the dataset.
- `self.model`: The model to be trained or evaluated.
- `self.optimizer`: The optimizer for training.
- `self._metrics`: Metrics used for evaluation.
- `self._callbacks`: Callbacks for training and evaluation.

#### Purpose:
Initializes the `Trainer` object with the provided configuration and mode, setting up the necessary components for training, evaluation, or testing.

---

### `register_callbacks(self, callbacks)`
#### Parameters:
- `callbacks` (list): A list of callback instances to be registered. Each callback should be an instance of a subclass of `Callback`.

#### Return Value:
- None

#### Purpose:
Registers additional callbacks for the training or evaluation process. Callbacks can be used for logging, checkpointing, and other purposes.

---

### `register_metrics(self, metrics)`
#### Parameters:
- `metrics` (list): A list of metric instances to be registered. Each metric should be an instance of a subclass of `Metric`.

#### Return Value:
- None

#### Purpose:
Registers additional metrics for evaluation. Metrics are used to assess the performance of the model during evaluation.

---

### `load_weights(self, weights, ARSL_eval=False)`
#### Parameters:
- `weights` (str): Path to the weights file to be loaded.
- `ARSL_eval` (bool): A flag indicating if ARSL evaluation is to be performed. Default is `False`.

#### Return Value:
- None

#### Purpose:
Loads the specified weights into the model. This is typically used to initialize the model with pre-trained weights.

---

### `load_weights_sde(self, det_weights, reid_weights)`
#### Parameters:
- `det_weights` (str): Path to the detection weights file.
- `reid_weights` (str): Path to the re-identification weights file.

#### Return Value:
- None

#### Purpose:
Loads the specified detection and re-identification weights into the model. This is used when the model has separate components for detection and re-identification.

---

### `resume_weights(self, weights)`
#### Parameters:
- `weights` (str): Path to the weights file to resume training from.

#### Return Value:
- None

#### Purpose:
Resumes training from the specified weights file. This is useful for continuing training after a pause or interruption.

---

### `train(self, validate=False)`
#### Parameters:
- `validate` (bool): A flag indicating if validation should be performed during training. Default is `False`.

#### Return Value:
- None

#### Purpose:
Starts the training process for the model. If validation is enabled, it will also evaluate the model on the validation dataset during training.

---

### `evaluate(self)`
#### Parameters:
- None

#### Return Value:
- None

#### Purpose:
Evaluates the model on the evaluation dataset. This method computes metrics and logs the evaluation results.

---

### `evaluate_slice(self, slice_size=[640, 640], overlap_ratio=[0.25, 0.25], combine_method='nms', match_threshold=0.6, match_metric='iou')`
#### Parameters:
- `slice_size` (list): Size of the slices for evaluation. Default is `[640, 640]`.
- `overlap_ratio` (list): Overlap ratio for the slices. Default is `[0.25, 0.25]`.
- `combine_method` (str): Method to combine results. Options are `'nms'` (Non-Maximum Suppression) or `'concat'`. Default is `'nms'`.
- `match_threshold` (float): Threshold for matching. Default is `0.6`.
- `match_metric` (str): Metric for matching. Default is `'iou'`.

#### Return Value:
- None

#### Purpose:
Evaluates the model using a sliding window approach with specified slice size and overlap. This is useful for processing large images.

---

### `slice_predict(self, images, slice_size=[640, 640], overlap_ratio=[0.25, 0.25], combine_method='nms', match_threshold=0.6, match_metric='iou', draw_threshold=0.5, output_dir='output', save_results=False, visualize=True)`
#### Parameters:
- `images` (list): List of images to predict.
- `slice_size` (list): Size of the slices for prediction. Default is `[640, 640]`.
- `overlap_ratio` (list): Overlap ratio for the slices. Default is `[0.25, 0.25]`.
- `combine_method` (str): Method to combine results. Options are `'nms'` or `'concat'`. Default is `'nms'`.
- `match_threshold` (float): Threshold for matching. Default is `0.6`.
- `match_metric` (str): Metric for matching. Default is `'iou'`.
- `draw_threshold` (float): Threshold for drawing results. Default is `0.5`.
- `output_dir` (str): Directory to save output results. Default is `'output'`.
- `save_results` (bool): A flag indicating if results should be saved. Default is `False`.
- `visualize` (bool): A flag indicating if results should be visualized. Default is `True`.

#### Return Value:
- None

#### Purpose:
Performs prediction on the provided images using a sliding window approach. The results can be saved and visualized based on the provided parameters.

---

### `predict(self, images, draw_threshold=0.5, output_dir='output', save_results=False, visualize=True, save_threshold=0)`
#### Parameters:
- `images` (list): List of images to predict.
- `draw_threshold` (float): Threshold for drawing results. Default is `0.5`.
- `output_dir` (str): Directory to save output results. Default is `'output'`.
- `save_results` (bool): A flag indicating if results should be saved. Default is `False`.
- `visualize` (bool): A flag indicating if results should be visualized. Default is `True`.
- `save_threshold` (float): Threshold for saving results. Default is `0`.

#### Return Value:
- list: A list of prediction results for the provided images.

#### Purpose:
Performs prediction on the provided images and returns the results. The results can be saved and visualized based on the provided parameters.

---

### `export(self, output_dir='output_inference', for_fd=False)`
#### Parameters:
- `output_dir` (str): Directory to save the exported model. Default is `'output_inference'`.
- `for_fd` (bool): A flag indicating if the model is for inference deployment. Default is `False`.

#### Return Value:
- None

#### Purpose:
Exports the trained model to a specified directory for inference. The model can be saved in a format suitable for deployment.

---

### `post_quant(self, output_dir='output_inference')`
#### Parameters:
- `output_dir` (str): Directory to save the post-quantized model. Default is `'output_inference'`.

#### Return Value:
- None

#### Purpose:
Performs post-training quantization on the model and saves the quantized model to the specified directory.

---

### `parse_mot_images(self, cfg)`
#### Parameters:
- `cfg` (dict): Configuration dictionary containing dataset settings.

#### Return Value:
- list: A list of image paths found in the specified dataset directory.

#### Purpose:
Parses and retrieves image paths from the specified dataset directory for evaluation or inference.

---

### `predict_culane(self, images, output_dir='output', save_results=False, visualize=True)`
#### Parameters:
- `images` (list): List of images to predict.
- `output_dir` (str): Directory to save output results. Default is `'output'`.
- `save_results` (bool): A flag indicating if results should be saved. Default is `False`.
- `visualize` (bool): A flag indicating if results should be visualized. Default is `True`.

#### Return Value:
- list: A list of prediction results for the provided images.

#### Purpose:
Performs lane detection prediction on the provided images and returns the results. The results can be saved and visualized based on the provided parameters.

---

### `reset_norm_param_attr(self, layer, **kwargs)`
#### Parameters:
- `layer` (nn.Layer): The layer whose normalization parameters are to be reset.
- `**kwargs`: Additional keyword arguments for normalization parameters.

#### Return Value:
- nn.Layer: The layer with reset normalization parameters.

#### Purpose:
Resets the normalization parameters of the specified layer, allowing for customization of attributes such as weight and bias. This is useful for adapting layers for different training scenarios.