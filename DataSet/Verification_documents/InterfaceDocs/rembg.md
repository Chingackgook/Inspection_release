# API Documentation

## Function: `remove()`

### Description
The `remove` function is designed to remove the background from an input image using a combination of deep learning models and traditional image processing techniques. It supports various input formats and provides options for alpha matting and background color customization.

### Parameters

- **data** (Union[bytes, PILImage, np.ndarray]): 
  - The input image data. It can be in the form of:
    - `bytes`: Raw image data.
    - `PILImage`: A PIL image object.
    - `np.ndarray`: A numpy array representation of the image.
  
- **alpha_matting** (bool, optional): 
  - Flag indicating whether to use alpha matting. Defaults to `False`.
  
- **alpha_matting_foreground_threshold** (int, optional): 
  - Foreground threshold for alpha matting. Valid range: 0-255. Defaults to `240`.
  
- **alpha_matting_background_threshold** (int, optional): 
  - Background threshold for alpha matting. Valid range: 0-255. Defaults to `10`.
  
- **alpha_matting_erode_size** (int, optional): 
  - Erosion size for alpha matting. Valid range: 0 or greater. Defaults to `10`.
  
- **session** (Optional[BaseSession], optional): 
  - A session object for the 'u2net' model. Defaults to `None`.
  
- **only_mask** (bool, optional): 
  - Flag indicating whether to return only the binary masks. Defaults to `False`.
  
- **post_process_mask** (bool, optional): 
  - Flag indicating whether to post-process the masks. Defaults to `False`.
  
- **bgcolor** (Optional[Tuple[int, int, int, int]], optional): 
  - Background color for the cutout image in RGBA format. Defaults to `None`.
  
- **force_return_bytes** (bool, optional): 
  - Flag indicating whether to return the cutout image as bytes. Defaults to `False`.
  
- **args** (Optional[Any]): 
  - Additional positional arguments.
  
- **kwargs** (Optional[Any]): 
  - Additional keyword arguments.

### Returns
- **Union[bytes, PILImage, np.ndarray]**: 
  - The cutout image with the background removed. The return type depends on the input type and the `force_return_bytes` flag:
    - If `data` is `bytes` or `force_return_bytes` is `True`, returns image data as `bytes`.
    - If `data` is a `PILImage`, returns a `PILImage` object.
    - If `data` is a `np.ndarray`, returns a numpy array representation of the image.

### Purpose
The `remove` function provides a high-precision background removal effect by leveraging deep learning models and traditional image processing techniques. It allows users to customize the output through various parameters, making it suitable for a wide range of applications in image processing and editing.

### Example Usage
```python
from pathlib import Path
from rembg import remove

# Configuration parameters
INPUT_IMAGE = "input.jpg"         # Input image path
OUTPUT_NAME = "output.png"        # Output file name
MODEL_NAME = "u2net"              # Optional model: u2netp, isnet-general-use, etc.

def main():
    try:
        # Check if input file exists
        input_path = Path(INPUT_IMAGE)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {INPUT_IMAGE} does not exist.")

        # Generate output path
        output_path = Path.cwd() / OUTPUT_NAME

        # Call remove to eliminate background
        print(f"⏳ Processing: {input_path.name} (Model: {MODEL_NAME})...")
        with open(input_path, "rb") as f_in:
            output_data = remove(f_in.read(), session_model=MODEL_NAME)

        # Save the result
        with open(output_path, "wb") as f_out:
            f_out.write(output_data)
        
        print(f"✅ Processing complete! Output saved to: {output_path}")

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
```

This documentation provides a comprehensive overview of the `remove` function, detailing its parameters, return values, and usage examples to assist developers in effectively utilizing the API.