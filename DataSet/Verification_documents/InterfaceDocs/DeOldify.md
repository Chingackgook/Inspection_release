# API Documentation

## Function: `get_watermarked`

### Description
Generates a watermarked version of a given PIL image. If an error occurs during the watermarking process, the original image is returned without any modifications.

### Parameters
- **pil_image** (`Image`): The input image in PIL format that needs to be watermarked.

### Returns
- **Image**: A PIL image that is either the watermarked version of the input image or the original image if an error occurs.

### Example
```python
watermarked_image = get_watermarked(original_image)
```

---

## Class: `ModelImageVisualizer`

### Description
The `ModelImageVisualizer` class provides a complete set of functionalities for image processing, visualization, and saving. It can be used to apply filters to images, visualize the results, and save the processed images.

### Attributes
- **filter** (`IFilter`): An instance of a filter that will be applied to the images.
- **results_dir** (`Path`): The directory where the processed images will be saved.

### Method: `__init__`

#### Description
Initializes a new instance of the `ModelImageVisualizer` class.

#### Parameters
- **filter** (`IFilter`): The filter to be used for processing images.
- **results_dir** (`str`, optional): The directory path where results will be saved. If `None`, defaults to the current directory.

### Returns
- **None**

### Example
```python
visualizer = ModelImageVisualizer(filter=my_filter, results_dir='output_directory')
```

---

### Method: `plot_transformed_image_from_url`

#### Description
Fetches an image from a given URL, applies the specified filter, and visualizes the transformed image. The result can be saved to a specified path.

#### Parameters
- **url** (`str`): The URL of the image to be processed.
- **path** (`str`, optional): The path where the original image will be saved. Defaults to 'test_images/image.png'.
- **results_dir** (`Path`, optional): The directory where the results will be saved. If `None`, uses the instance's `results_dir`.
- **figsize** (`Tuple[int, int]`, optional): The size of the figure for visualization. Defaults to (20, 20).
- **render_factor** (`int`, optional): A factor that influences the rendering quality. If `None`, defaults to the filter's internal settings.
- **display_render_factor** (`bool`, optional): Whether to display the render factor on the plot. Defaults to `False`.
- **compare** (`bool`, optional): Whether to compare the original and transformed images side by side. Defaults to `False`.
- **post_process** (`bool`, optional): Whether to apply post-processing to the transformed image. Defaults to `True`.
- **watermarked** (`bool`, optional): Whether to apply a watermark to the transformed image. Defaults to `True`.

### Returns
- **Path**: The path where the transformed image is saved.

### Example
```python
result_path = visualizer.plot_transformed_image_from_url('http://example.com/image.jpg')
```

---

### Method: `plot_transformed_image`

#### Description
Loads an image from a specified path, applies the filter, and visualizes the transformed image. The result can be saved to a specified directory.

#### Parameters
- **path** (`str`): The path of the image to be processed.
- **results_dir** (`Path`, optional): The directory where the results will be saved. If `None`, uses the instance's `results_dir`.
- **figsize** (`Tuple[int, int]`, optional): The size of the figure for visualization. Defaults to (20, 20).
- **render_factor** (`int`, optional): A factor that influences the rendering quality. If `None`, defaults to the filter's internal settings.
- **display_render_factor** (`bool`, optional): Whether to display the render factor on the plot. Defaults to `False`.
- **compare** (`bool`, optional): Whether to compare the original and transformed images side by side. Defaults to `False`.
- **post_process** (`bool`, optional): Whether to apply post-processing to the transformed image. Defaults to `True`.
- **watermarked** (`bool`, optional): Whether to apply a watermark to the transformed image. Defaults to `True`.

### Returns
- **Path**: The path where the transformed image is saved.

### Example
```python
result_path = visualizer.plot_transformed_image('path/to/image.png')
```

---

### Method: `get_transformed_image`

#### Description
Applies the specified filter to an image located at a given path and returns the transformed image. Optionally applies post-processing and watermarking.

#### Parameters
- **path** (`Path`): The path of the image to be processed.
- **render_factor** (`int`, optional): A factor that influences the rendering quality. If `None`, defaults to the filter's internal settings.
- **post_process** (`bool`, optional): Whether to apply post-processing to the transformed image. Defaults to `True`.
- **watermarked** (`bool`, optional): Whether to apply a watermark to the transformed image. Defaults to `True`.

### Returns
- **Image**: The transformed PIL image after applying the filter and any optional processing.

### Example
```python
transformed_image = visualizer.get_transformed_image(Path('path/to/image.png'))
```