# API Documentation

## Class: Net

### Description
The `Net` class is a PyTorch neural network model for background removal using various U2Net architectures. It initializes the model based on the specified architecture and provides a forward method for inference.

### Attributes
- `net`: The loaded U2Net model.

### Method: __init__

#### Description
Initializes the `Net` class by loading the specified U2Net model.

#### Parameters
- `model_name` (str): The name of the model to load. Options are:
  - `"u2netp"`
  - `"u2net"`
  - `"u2net_human_seg"`

#### Return Value
None

#### Purpose
To create an instance of the `Net` class with the specified U2Net model loaded and ready for inference.

---

### Method: forward

#### Description
Performs a forward pass through the network to predict the alpha mask for the input image.

#### Parameters
- `block_input` (torch.Tensor): A tensor of shape (N, H, W, C) where N is the number of images, H is the height, W is the width, and C is the number of channels (3 for RGB).

#### Return Value
torch.Tensor: A tensor of shape (N, H, W) representing the predicted alpha mask for each input image.

#### Purpose
To generate an alpha mask from the input images, which can be used for background removal.

---

## Function: alpha_matting_cutout

### Description
Applies alpha matting to create a cutout image from the input image and its corresponding mask.

#### Parameters
- `img` (PIL.Image): The input image from which to create a cutout.
- `mask` (PIL.Image): The mask image used to determine the foreground and background.
- `foreground_threshold` (int): Threshold value to determine foreground pixels (0-255).
- `background_threshold` (int): Threshold value to determine background pixels (0-255).
- `erode_structure_size` (int): Size of the structuring element for erosion (0 or greater).
- `base_size` (int): The base size to which the image will be resized (greater than 0).

#### Return Value
PIL.Image: The cutout image with the background removed.

#### Purpose
To create a refined cutout of the foreground object using alpha matting techniques.

---

## Function: naive_cutout

### Description
Creates a simple cutout image by compositing the input image with a transparent background using the provided mask.

#### Parameters
- `img` (PIL.Image): The input image from which to create a cutout.
- `mask` (PIL.Image): The mask image used to determine the foreground.

#### Return Value
PIL.Image: The cutout image with the background removed.

#### Purpose
To create a basic cutout of the foreground object without advanced alpha matting.

---

## Function: get_model

### Description
Loads the specified U2Net model for background removal.

#### Parameters
- `model_name` (str): The name of the model to load. Options are:
  - `"u2netp"`
  - `"u2net_human_seg"`
  - `"u2net"`

#### Return Value
torch.nn.Module: The loaded U2Net model.

#### Purpose
To retrieve the appropriate U2Net model based on the specified name for background removal tasks.

---

## Function: remove

### Description
Removes the background from an input image using the specified model and optional alpha matting.

#### Parameters
- `data` (np.ndarray or bytes): The input image data as a NumPy array or bytes.
- `model_name` (str): The name of the model to use for background removal. Default is `"u2net"`.
- `alpha_matting` (bool): Whether to apply alpha matting. Default is False.
- `alpha_matting_foreground_threshold` (int): Foreground threshold for alpha matting (0-255). Default is 240.
- `alpha_matting_background_threshold` (int): Background threshold for alpha matting (0-255). Default is 10.
- `alpha_matting_erode_structure_size` (int): Erosion structure size for alpha matting (0 or greater). Default is 10.
- `alpha_matting_base_size` (int): Base size for resizing the image (greater than 0). Default is 1000.

#### Return Value
bytes: The cutout image in PNG format as a byte buffer.

#### Purpose
To remove the background from the input image and return the resulting cutout image.

---

## Function: iter_frames

### Description
Iterates through the frames of a video file, resizing each frame to a specified height.

#### Parameters
- `path` (str): The file path to the video.

#### Return Value
generator: A generator that yields frames as NumPy arrays.

#### Purpose
To provide an iterable over the frames of a video, allowing for processing of each frame individually.

---

## Function: remove_many

### Description
Removes backgrounds from multiple images in a batch using the specified neural network.

#### Parameters
- `image_data` (List[np.ndarray]): A list of images as NumPy arrays.
- `net` (Net): An instance of the `Net` class containing the loaded model.

#### Return Value
np.ndarray: A NumPy array containing the alpha masks for each input image.

#### Purpose
To efficiently process and remove backgrounds from multiple images in a single batch operation.

