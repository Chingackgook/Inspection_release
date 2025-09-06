# API Documentation

## Function: `load_llama3_tokenizer`

### Description
Loads the tokenizer for the Llama-3.2-1B model from Hugging Face's model hub. This tokenizer is configured to handle special tokens for text processing.

### Returns
- **AutoTokenizer**: An instance of the `AutoTokenizer` configured for the Llama-3.2-1B model.

---

## Class: `Generator`

### Description
The `Generator` class is responsible for generating audio from text input using a specified model. It handles the tokenization of both text and audio segments and manages the generation process.

### Attributes
- **_model**: An instance of `Model` that represents the underlying model used for generation.
- **_text_tokenizer**: An instance of `AutoTokenizer` for tokenizing text input.
- **_audio_tokenizer**: An instance of the audio tokenizer for encoding audio input.
- **_watermarker**: An instance of the watermarker used to apply a watermark to generated audio.
- **sample_rate**: The sample rate of the audio generated.
- **device**: The device (CPU or GPU) on which the model is loaded.

### Method: `__init__`

#### Parameters
- **model** (`Model`): The model instance used for audio generation.

#### Description
Initializes the `Generator` class, setting up the model, tokenizers, and watermarker. It also configures the device for computation.

---

### Method: `generate`

#### Parameters
- **text** (`str`): The text input to be converted into audio. 
  - **Value Range**: Any string of text.
  
- **speaker** (`int`): An identifier for the speaker.
  - **Value Range**: Non-negative integer representing the speaker ID.
  
- **context** (`List[Segment]`): A list of context segments that provide additional information for generation.
  - **Value Range**: List of `Segment` objects.
  
- **max_audio_length_ms** (`float`, optional): The maximum length of the generated audio in milliseconds. Default is 90,000 ms (90 seconds).
  - **Value Range**: Positive float value.
  
- **temperature** (`float`, optional): Controls the randomness of the generation. Higher values result in more random outputs. Default is 0.9.
  - **Value Range**: Float value, typically between 0 and 1.
  
- **topk** (`int`, optional): The number of top tokens to consider during generation. Default is 50.
  - **Value Range**: Positive integer.

#### Returns
- **torch.Tensor**: A tensor representing the generated audio waveform.

#### Description
Generates audio from the provided text input and context segments. It tokenizes the input, manages the generation process, and applies a watermark to the resulting audio to identify it as AI-generated.

---

## Function: `load_csm_1b`

### Parameters
- **device** (`str`, optional): The device to load the model onto. Default is "cuda".
  - **Value Range**: "cuda" for GPU or "cpu" for CPU.

### Returns
- **Generator**: An instance of the `Generator` class initialized with the CSM-1B model.

### Description
Loads the CSM-1B model and returns a `Generator` instance configured to generate audio using this model. The model is transferred to the specified device and set to use bfloat16 precision for efficient computation.