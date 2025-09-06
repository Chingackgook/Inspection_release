# API Documentation

## Function: `parse_range`

### Description
Parses a comma-separated list of numbers or ranges and returns a list of integers.

### Parameters
- **s** (Union[str, List]): A string representing a list of numbers or ranges (e.g., `'1,2,5-10'`) or a list of integers.
  - **Value Range**: The string can contain individual numbers or ranges in the format `start-end`.
  
### Returns
- **List[int]**: A list of integers parsed from the input string or list.

### Example
- Input: `'1,2,5-10'`
- Output: `[1, 2, 5, 6, 7, 8, 9, 10]`

---

## Function: `parse_vec2`

### Description
Parses a floating-point 2-vector from a string of the syntax 'a,b'.

### Parameters
- **s** (Union[str, Tuple[float, float]]): A string representing a 2D vector (e.g., `'0,1'`) or a tuple of two floats.
  - **Value Range**: The string must contain exactly two float values separated by a comma.

### Returns
- **Tuple[float, float]**: A tuple containing two floating-point numbers parsed from the input string.

### Example
- Input: `'0,1'`
- Output: `(0.0, 1.0)`

---

## Function: `make_transform`

### Description
Creates a transformation matrix for translation and rotation in 2D space.

### Parameters
- **translate** (Tuple[float, float]): A tuple representing the translation in the X and Y coordinates.
  - **Value Range**: Any floating-point values.
- **angle** (float): The rotation angle in degrees.
  - **Value Range**: Any floating-point value.

### Returns
- **np.ndarray**: A 3x3 transformation matrix for the specified translation and rotation.

### Example
- Input: `translate=(1, 2), angle=45`
- Output: A 3x3 transformation matrix representing the specified translation and rotation.

---

## Function: `generate_images`

### Description
Generates images using a pretrained StyleGAN model based on specified parameters.

### Parameters
- **network_pkl** (str): The URL or path to the network pickle file containing the pretrained model.
  - **Value Range**: Must be a valid URL or file path.
- **seeds** (List[int]): A list of random seeds for generating images (e.g., `[0, 1, 4]`).
  - **Value Range**: Must be a list of integers.
- **truncation_psi** (float): The truncation psi value for image generation.
  - **Value Range**: Typically between 0 and 1.
- **noise_mode** (str): The mode of noise to use during generation. Options are `'const'`, `'random'`, or `'none'`.
  - **Value Range**: Must be one of the specified choices.
- **outdir** (str): The directory where the generated images will be saved.
  - **Value Range**: Must be a valid directory path.
- **translate** (Tuple[float, float]): A tuple representing the translation in the X and Y coordinates.
  - **Value Range**: Any floating-point values.
- **rotate** (float): The rotation angle in degrees.
  - **Value Range**: Any floating-point value.
- **class_idx** (Optional[int]): The class label for conditional generation. If not specified, unconditional generation is performed.
  - **Value Range**: Must be an integer or None.

### Returns
- **None**: This function does not return a value. It saves generated images to the specified output directory.

### Example
```bash
python gen_images.py --outdir=out --trunc=1 --seeds=2 \
    --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl
```

### Notes
- The function will create the output directory if it does not exist.
- It will generate images based on the specified seeds and parameters, saving them in the specified output directory.