# API Documentation

## Class: Model

The `Model` class implements a neural network model for optical flow estimation. It provides methods for training, evaluation, inference, and model management.

### Attributes:
- `flownet`: The neural network model, either an instance of `IFNet` or `IFNet_m`.
- `optimG`: The optimizer used for training the model, specifically AdamW.
- `epe`: An instance of the EPE (End Point Error) loss function.
- `lap`: An instance of the Laplacian loss function.
- `sobel`: An instance of the Sobel operator for edge detection.

### Method: `__init__(self, local_rank=-1, arbitrary=False)`

#### Parameters:
- `local_rank` (int, optional): The rank of the process in a distributed training setup. Default is -1, indicating single GPU training.
- `arbitrary` (bool, optional): If True, initializes the model with `IFNet_m`; otherwise, initializes with `IFNet`. Default is False.

#### Return Value:
- None

#### Description:
Initializes the `Model` class, sets up the neural network, optimizer, and loss functions. It also configures the model for distributed training if `local_rank` is specified.

---

### Method: `train(self)`

#### Parameters:
- None

#### Return Value:
- None

#### Description:
Sets the model to training mode, enabling dropout and batch normalization layers.

---

### Method: `eval(self)`

#### Parameters:
- None

#### Return Value:
- None

#### Description:
Sets the model to evaluation mode, disabling dropout and batch normalization layers.

---

### Method: `device(self)`

#### Parameters:
- None

#### Return Value:
- None

#### Description:
Moves the model to the appropriate device (GPU or CPU) based on availability.

---

### Method: `load_model(self, path, rank=0)`

#### Parameters:
- `path` (str): The file path from which to load the model weights.
- `rank` (int, optional): The rank of the process in a distributed training setup. Default is 0.

#### Return Value:
- None

#### Description:
Loads the model weights from the specified path. If the model was saved in a distributed manner, it converts the parameter names to match the current model structure.

---

### Method: `save_model(self, path, rank=0)`

#### Parameters:
- `path` (str): The file path where the model weights will be saved.
- `rank` (int, optional): The rank of the process in a distributed training setup. Default is 0.

#### Return Value:
- None

#### Description:
Saves the model weights to the specified path. This method is only executed by the process with rank 0 in a distributed setup.

---

### Method: `inference(self, img0, img1, scale=1, scale_list=None, TTA=False, timestep=0.5)`

#### Parameters:
- `img0` (Tensor): The first input image tensor.
- `img1` (Tensor): The second input image tensor.
- `scale` (int, optional): The scale factor for resizing images. Default is 1.
- `scale_list` (list, optional): A list of scale factors for the model. Default is [4, 2, 1].
- `TTA` (bool, optional): If True, applies test-time augmentation. Default is False.
- `timestep` (float, optional): The timestep for the model's inference. Default is 0.5.

#### Return Value:
- Tensor: The merged output image after processing.

#### Description:
Performs inference on the two input images to estimate optical flow. Optionally applies test-time augmentation if `TTA` is set to True.

---

### Method: `update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None)`

#### Parameters:
- `imgs` (Tensor): A tensor containing the input images, concatenated along the channel dimension.
- `gt` (Tensor): The ground truth tensor for loss calculation.
- `learning_rate` (float, optional): The learning rate for the optimizer. Default is 0.
- `mul` (int, optional): A multiplier for the loss. Default is 1.
- `training` (bool, optional): If True, the model is set to training mode; otherwise, it is set to evaluation mode. Default is True.
- `flow_gt` (Tensor, optional): Ground truth flow tensor, if available. Default is None.

#### Return Value:
- Tuple: A tuple containing the merged output image and a dictionary with additional outputs including masks, flows, and losses.

#### Description:
Updates the model parameters based on the input images and ground truth. Computes the losses and performs a backward pass if in training mode. Returns the merged output and relevant metrics.

