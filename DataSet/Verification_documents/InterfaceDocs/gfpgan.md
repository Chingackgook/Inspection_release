# API Documentation

## Class: GFPGANer

### Description
The `GFPGANer` class is a helper for restoring images using the GFPGAN model. It detects and crops faces from input images, resizes them to 512x512 pixels, and applies the GFPGAN model to restore the faces. The background of the image can also be enhanced using a specified upsampler. Finally, the restored faces are pasted back onto the enhanced background.

### Attributes
- **upscale (float)**: The upscale factor for the final output. Default is 2. This value determines how much the final image will be enlarged.
- **bg_upsampler (nn.Module)**: The upsampler for the background. Default is None. This should be a neural network module capable of enhancing the background.
- **device (torch.device)**: The device on which the model will run (CPU or GPU).
- **gfpgan (nn.Module)**: The GFPGAN model used for face restoration.
- **face_helper (FaceRestoreHelper)**: An instance of the `FaceRestoreHelper` class that assists in face detection and alignment.

### Methods

#### `__init__(self, model_path, upscale=2, arch='clean', channel_multiplier=2, bg_upsampler=None, device=None)`

**Description**: Initializes the `GFPGANer` class, loading the specified GFPGAN model and setting up the necessary parameters.

**Parameters**:
- `model_path (str)`: The path to the GFPGAN model. This can be a local path or a URL. 
- `upscale (float)`: The upscale factor for the final output. Default is 2. (Range: > 0)
- `arch (str)`: The architecture of the GFPGAN model to use. Options are 'clean', 'original', 'bilinear', or 'RestoreFormer'. Default is 'clean'.
- `channel_multiplier (int)`: The channel multiplier for large networks of StyleGAN2. Default is 2. (Range: > 0)
- `bg_upsampler (nn.Module)`: The upsampler for the background. Default is None.
- `device (torch.device)`: The device to run the model on. If None, it defaults to 'cuda' if available, otherwise 'cpu'.

**Return Value**: None

---

#### `enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5)`

**Description**: Enhances the input image by detecting and restoring faces. It can also upscale the background and paste the restored faces back onto it.

**Parameters**:
- `img (numpy.ndarray)`: The input image to be enhanced. It should be in BGR format.
- `has_aligned (bool)`: Indicates if the input image has already been aligned. Default is False.
- `only_center_face (bool)`: If True, only the center face will be processed. Default is False.
- `paste_back (bool)`: If True, the restored faces will be pasted back onto the upsampled background. Default is True.
- `weight (float)`: The blending weight for the restored face. Default is 0.5. (Range: 0.0 to 1.0)

**Return Value**: A tuple containing:
- `cropped_faces (list)`: A list of cropped face images.
- `restored_faces (list)`: A list of restored face images.
- `restored_img (numpy.ndarray)`: The final restored image with faces pasted back, or None if `paste_back` is False.

---

This documentation provides a comprehensive overview of the `GFPGANer` class, its initializer, and the `enhance` method, detailing their parameters, return values, and purposes.