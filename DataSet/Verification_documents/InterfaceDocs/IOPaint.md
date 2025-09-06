# API Documentation

## Function: `glob_images`

### Description
The `glob_images` function retrieves image files from a specified path. It supports both individual files and directories, returning a dictionary of image file names (without extensions) mapped to their respective file paths.

### Parameters
- **path** (`Path`): The path to a file or directory to search for image files.
  - **Value Range**: Can be a single file path or a directory path.

### Returns
- **Dict[str, Path]**: A dictionary where the keys are the names of the image files (without extensions) and the values are the corresponding `Path` objects.

### Purpose
This function is used to gather image files from a specified location, making it easier to process them in batch operations.

---

## Function: `batch_inpaint`

### Description
The `batch_inpaint` function performs inpainting on a batch of images using a specified model. It takes images and their corresponding masks, processes them, and saves the inpainted results to an output directory.

### Parameters
- **model** (`str`): The name of the inpainting model to use.
  - **Value Range**: Must be a valid model name recognized by the `ModelManager`.

- **device**: The device on which the model will run (e.g., CPU or GPU).
  - **Value Range**: Typically a string indicating the device type (e.g., "cpu", "cuda").

- **image** (`Path`): The path to a directory containing input images or a single image file.
  - **Value Range**: Can be a directory path or a file path.

- **mask** (`Path`): The path to a directory containing mask images or a single mask file.
  - **Value Range**: Can be a directory path or a file path.

- **output** (`Path`): The path to the directory where the output images will be saved.
  - **Value Range**: Must be a directory path.

- **config** (`Optional[Path]`): An optional path to a configuration file for inpainting settings.
  - **Value Range**: Can be a valid file path or `None`.

- **concat** (`bool`): A flag indicating whether to concatenate the original image, mask, and inpainted result horizontally.
  - **Value Range**: `True` or `False`.

### Returns
- **None**: This function does not return a value. It saves the inpainted images to the specified output directory.

### Purpose
The `batch_inpaint` function is designed to automate the inpainting process for multiple images, allowing users to specify models, masks, and configurations, and to save the results efficiently. It also provides progress tracking and memory usage information during processing.

