# API Documentation for SparkTTS

## Class: SparkTTS

The `SparkTTS` class implements the core functionality for text-to-speech (TTS) generation and intelligent speech synthesis. It provides a unified and user-friendly interface for generating speech from text.

### Attributes:
- **device (torch.device)**: The device (CPU or GPU) on which the model runs.
- **model_dir (Path)**: The directory containing the model and configuration files.
- **configs (dict)**: Configuration settings loaded from the model's config file.
- **sample_rate (int)**: The sample rate for audio generation.
- **tokenizer (AutoTokenizer)**: The tokenizer used for processing text input.
- **model (AutoModelForCausalLM)**: The causal language model used for generating speech.
- **audio_tokenizer (BiCodecTokenizer)**: The audio tokenizer for converting audio tokens to waveforms.

### Method: `__init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0"))`
Initializes the `SparkTTS` model with the provided configurations and device.

#### Parameters:
- **model_dir (Path)**: Directory containing the model and config files.
- **device (torch.device)**: The device (CPU/GPU) to run the model on. Default is `torch.device("cuda:0")`.

#### Return Value:
- None

#### Purpose:
To set up the TTS model and load necessary configurations for inference.

---

### Method: `process_prompt(self, text: str, prompt_speech_path: Path, prompt_text: str = None) -> Tuple[str, torch.Tensor]`
Processes the input for voice cloning.

#### Parameters:
- **text (str)**: The text input to be converted to speech.
- **prompt_speech_path (Path)**: Path to the audio file used as a prompt.
- **prompt_text (str, optional)**: Transcript of the prompt audio. Default is `None`.

#### Return Value:
- **Tuple[str, torch.Tensor]**: A tuple containing the input prompt as a string and global token IDs as a tensor.

#### Purpose:
To prepare the input tokens for the model by tokenizing the prompt audio and combining it with the text input.

---

### Method: `process_prompt_control(self, gender: str, pitch: str, speed: str, text: str) -> str`
Processes the input for voice creation with controllable attributes.

#### Parameters:
- **gender (str)**: Gender of the voice. Must be either "female" or "male".
- **pitch (str)**: Pitch level of the voice. Must be one of: "very_low", "low", "moderate", "high", "very_high".
- **speed (str)**: Speed of the speech. Must be one of: "very_low", "low", "moderate", "high", "very_high".
- **text (str)**: The text input to be converted to speech.

#### Return Value:
- **str**: The input prompt as a string.

#### Purpose:
To create a prompt for the TTS model that includes controllable attributes such as gender, pitch, and speed.

---

### Method: `inference(self, text: str, prompt_speech_path: Path = None, prompt_text: str = None, gender: str = None, pitch: str = None, speed: str = None, temperature: float = 0.8, top_k: float = 50, top_p: float = 0.95) -> torch.Tensor`
Performs inference to generate speech from text, incorporating prompt audio and/or text.

#### Parameters:
- **text (str)**: The text input to be converted to speech.
- **prompt_speech_path (Path, optional)**: Path to the audio file used as a prompt. Default is `None`.
- **prompt_text (str, optional)**: Transcript of the prompt audio. Default is `None`.
- **gender (str, optional)**: Gender of the voice. Must be either "female" or "male". Default is `None`.
- **pitch (str, optional)**: Pitch level of the voice. Must be one of: "very_low", "low", "moderate", "high", "very_high". Default is `None`.
- **speed (str, optional)**: Speed of the speech. Must be one of: "very_low", "low", "moderate", "high", "very_high". Default is `None`.
- **temperature (float, optional)**: Sampling temperature for controlling randomness. Default is `0.8`.
- **top_k (float, optional)**: Top-k sampling parameter. Default is `50`.
- **top_p (float, optional)**: Top-p (nucleus) sampling parameter. Default is `0.95`.

#### Return Value:
- **torch.Tensor**: Generated waveform as a tensor.

#### Purpose:
To generate speech from the provided text input, optionally using prompt audio and/or controllable attributes. The method utilizes the model to produce a waveform tensor representing the synthesized speech.

