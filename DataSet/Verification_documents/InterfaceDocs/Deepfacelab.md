# API Documentation

## Function: `process_folder`

### Description
The `process_folder` function enhances a faceset located in a specified directory. It processes images using either CPU or GPU resources, depending on the configuration provided. The enhanced images are saved in a new directory, and the user has the option to merge the enhanced images back into the original directory.

### Parameters
- **dirpath** (`Path`): 
  - **Description**: The path to the directory containing the images to be enhanced.
  - **Value Range**: Must be a valid directory path that exists on the filesystem.

- **cpu_only** (`bool`, optional): 
  - **Description**: A flag indicating whether to use only CPU for processing. If set to `True`, the function will not utilize any GPU resources.
  - **Value Range**: `True` or `False`. Default is `False`.

- **force_gpu_idxs** (`list[int]`, optional): 
  - **Description**: A list of GPU indices to be used for processing. If provided, the function will use these specific GPUs instead of prompting the user for selection.
  - **Value Range**: A list of integers representing valid GPU indices. Default is `None`, which prompts the user for GPU selection.

### Return Value
- **Return Type**: `None`
- **Description**: The function does not return any value. It performs file operations and logging as side effects.

### Purpose
The `process_folder` function is designed to enhance images in a specified directory by applying a faceset enhancement algorithm. It manages the creation of an output directory for enhanced images, handles the cleanup of existing enhanced images, and provides an option to merge the enhanced images back into the original directory.

### Example Usage
```python
from pathlib import Path

# Specify the directory containing images
image_directory = Path('/path/to/images')

# Process the folder with default settings
process_folder(image_directory)

# Process the folder using only CPU
process_folder(image_directory, cpu_only=True)

# Process the folder using specific GPU indices
process_folder(image_directory, force_gpu_idxs=[0, 1])
```

### Notes
- The function utilizes logging to provide feedback on the processing status and any actions taken, such as merging files or removing directories.
- Error handling is implemented during the file copy operation to ensure that the process continues even if some files cannot be copied.

# API Documentation

## Function: `extract_video`

### Description
The `extract_video` function extracts frames from a video file and saves them as images in a specified directory. The user can specify the output image format and the frames per second (FPS) to control how many frames are extracted.

### Parameters
- **input_file** (`str`): 
  - **Description**: The path to the input video file from which frames will be extracted.
  - **Value Range**: Must be a valid video file path.

- **output_dir** (`str`): 
  - **Description**: The directory where the extracted frames will be saved.
  - **Value Range**: Must be a valid directory path.

- **output_ext** (`str`, optional): 
  - **Description**: The desired image format for the extracted frames (e.g., 'png', 'jpg').
  - **Value Range**: Must be one of the supported formats, default is `None`, which prompts the user for input.

- **fps** (`int`, optional): 
  - **Description**: The number of frames to extract per second. If set to 0, all frames will be extracted.
  - **Value Range**: Non-negative integer. Default is `None`, which prompts the user for input.

### Return Value
- **Return Type**: `None`
- **Description**: The function does not return any value. It performs file operations and logging as side effects.

### Purpose
The `extract_video` function is designed to extract frames from a video file and save them as images in a specified format and directory. It allows users to control the extraction rate and format of the output images.

---

## Function: `cut_video`

### Description
The `cut_video` function creates a new video file by cutting a specified segment from an input video file. The user can specify the start and end times for the cut, as well as the audio track and bitrate.

### Parameters
- **input_file** (`str`): 
  - **Description**: The path to the input video file to be cut.
  - **Value Range**: Must be a valid video file path.

- **from_time** (`str`, optional): 
  - **Description**: The start time for the cut in the format "HH:MM:SS.mmm".
  - **Value Range**: Must be a valid time string. Default is `None`, which prompts the user for input.

- **to_time** (`str`, optional): 
  - **Description**: The end time for the cut in the format "HH:MM:SS.mmm".
  - **Value Range**: Must be a valid time string. Default is `None`, which prompts the user for input.

- **audio_track_id** (`int`, optional): 
  - **Description**: The ID of the audio track to include in the output video.
  - **Value Range**: Non-negative integer. Default is `None`, which prompts the user for input.

- **bitrate** (`int`, optional): 
  - **Description**: The bitrate for the output video in MB/s.
  - **Value Range**: Positive integer. Default is `None`, which prompts the user for input.

### Return Value
- **Return Type**: `None`
- **Description**: The function does not return any value. It performs file operations and logging as side effects.

### Purpose
The `cut_video` function allows users to create a new video file by extracting a specific segment from an existing video. It provides options for audio track selection and bitrate configuration.

---

## Function: `denoise_image_sequence`

### Description
The `denoise_image_sequence` function applies a denoising filter to a sequence of images in a specified directory. The user can specify the denoise factor, which controls the strength of the denoising effect.

### Parameters
- **input_dir** (`str`): 
  - **Description**: The path to the directory containing the images to be denoised.
  - **Value Range**: Must be a valid directory path.

- **ext** (`str`, optional): 
  - **Description**: The file extension of the images to be processed.
  - **Value Range**: Must be a valid image file extension. Default is `None`.

- **factor** (`int`, optional): 
  - **Description**: The strength of the denoising effect, ranging from 1 to 20.
  - **Value Range**: Integer between 1 and 20. Default is `None`, which prompts the user for input.

### Return Value
- **Return Type**: `None`
- **Description**: The function does not return any value. It performs file operations and logging as side effects.

### Purpose
The `denoise_image_sequence` function is designed to reduce noise in a sequence of images by applying a denoising filter. It allows users to specify the strength of the denoising effect.

---

## Function: `video_from_sequence`

### Description
The `video_from_sequence` function creates a video file from a sequence of images located in a specified directory. The user can specify the output file, frame rate, bitrate, and whether to include audio.

### Parameters
- **input_dir** (`str`): 
  - **Description**: The path to the directory containing the images to be converted into a video.
  - **Value Range**: Must be a valid directory path.

- **output_file** (`str`): 
  - **Description**: The path to the output video file.
  - **Value Range**: Must be a valid file path.

- **reference_file** (`str`, optional): 
  - **Description**: The path to a reference video file to obtain properties such as frame rate and audio track.
  - **Value Range**: Must be a valid video file path. Default is `None`.

- **ext** (`str`, optional): 
  - **Description**: The file extension of the input images.
  - **Value Range**: Must be a valid image file extension. Default is `None`, which prompts the user for input.

- **fps** (`int`, optional): 
  - **Description**: The frame rate for the output video.
  - **Value Range**: Positive integer. Default is `None`, which prompts the user for input.

- **bitrate** (`int`, optional): 
  - **Description**: The bitrate for the output video in MB/s.
  - **Value Range**: Positive integer. Default is `None`, which prompts the user for input.

- **include_audio** (`bool`, optional): 
  - **Description**: A flag indicating whether to include audio from the reference file.
  - **Value Range**: `True` or `False`. Default is `False`.

- **lossless** (`bool`, optional): 
  - **Description**: A flag indicating whether to use a lossless codec for the output video.
  - **Value Range**: `True` or `False`. Default is `None`, which prompts the user for input.

### Return Value
- **Return Type**: `None`
- **Description**: The function does not return any value. It performs file operations and logging as side effects.

### Purpose
The `video_from_sequence` function allows users to create a video file from a sequence of images, with options for frame rate, bitrate, and audio inclusion. It can also use a reference video to inherit properties.

