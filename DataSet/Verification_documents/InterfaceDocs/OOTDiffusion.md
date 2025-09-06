# API Documentation for OOTDiffusionHD

## Class: OOTDiffusionHD

### Description
The `OOTDiffusionHD` class is designed for high-definition image generation using diffusion models. It utilizes various pretrained models for image processing and text encoding, allowing for the generation of images based on provided prompts and images.

### Attributes
- **gpu_id**: (str) The identifier for the GPU to be used for computations (e.g., 'cuda:0').
- **pipe**: (OotdPipeline) The main pipeline for generating images, initialized with various pretrained models.
- **auto_processor**: (AutoProcessor) A processor for handling input images.
- **image_encoder**: (CLIPVisionModelWithProjection) A model for encoding images into embeddings.
- **tokenizer**: (CLIPTokenizer) A tokenizer for processing text prompts.
- **text_encoder**: (CLIPTextModel) A model for encoding text prompts into embeddings.

### Method: `__init__(self, gpu_id)`

#### Description
Initializes the `OOTDiffusionHD` class by loading the necessary pretrained models and setting up the pipeline.

#### Parameters
- **gpu_id** (int): The ID of the GPU to be used for computations. This should be a non-negative integer representing the GPU index.

#### Return Value
- None

---

### Method: `tokenize_captions(self, captions, max_length)`

#### Description
Tokenizes the provided captions into input IDs suitable for the text encoder.

#### Parameters
- **captions** (list of str): A list of captions to be tokenized.
- **max_length** (int): The maximum length of the tokenized captions. This should be a positive integer.

#### Return Value
- (torch.Tensor) A tensor containing the tokenized input IDs.

---

### Method: `__call__(self, model_type='hd', category='upperbody', image_garm=None, image_vton=None, mask=None, image_ori=None, num_samples=1, num_steps=20, image_scale=1.0, seed=-1)`

#### Description
Generates images based on the provided parameters, including model type, category, and input images.

#### Parameters
- **model_type** (str): The type of model to use for generation. Acceptable values are 'hd' for high-definition or 'dc' for diffusion conditioning. Default is 'hd'.
- **category** (str): The category of the image to be generated (e.g., 'upperbody'). Default is 'upperbody'.
- **image_garm** (PIL.Image or np.ndarray): The garment image to be used in the generation process. Default is None.
- **image_vton** (PIL.Image or np.ndarray): The virtual try-on image. Default is None.
- **mask** (PIL.Image or np.ndarray): An optional mask image to guide the generation. Default is None.
- **image_ori** (PIL.Image or np.ndarray): The original image to be modified. Default is None.
- **num_samples** (int): The number of images to generate. Must be a positive integer. Default is 1.
- **num_steps** (int): The number of inference steps to take during generation. Must be a positive integer. Default is 20.
- **image_scale** (float): The scale for image guidance. Must be a positive float. Default is 1.0.
- **seed** (int): The random seed for reproducibility. If set to -1, a random seed will be generated. Default is -1.

#### Return Value
- (list of PIL.Image) A list of generated images.

#### Example Usage
```python
oot_diffusion = OOTDiffusionHD(gpu_id=0)
generated_images = oot_diffusion(model_type='hd', category='upperbody', image_garm=some_image, num_samples=5)
```

### Notes
- Ensure that the input images are in the correct format (PIL Image or NumPy array).
- The method will raise a `ValueError` if `model_type` is not 'hd' or 'dc'.

