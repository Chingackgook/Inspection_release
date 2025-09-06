# API Documentation

## Function: `speech_to_text`

### Description
The `speech_to_text` function transcribes audio files into text using the Whisper model. It supports various languages and can handle audio files from local paths or URLs. The function processes the audio, converts it to the required format, and generates a subtitle file in SRT format.

### Parameters

- **model_name** (`str`, optional): 
  - Description: The name of the Whisper model to use for transcription.
  - Default: `'large-v2'`
  - Value Range: Must be a valid model name supported by the Whisper library.

- **language** (`str`, optional): 
  - Description: The language of the audio file. If set to "auto", the function will attempt to detect the language automatically.
  - Default: `"auto"`
  - Value Range: A two-letter language code (e.g., 'en' for English, 'zh' for Chinese) or "auto".

- **prompt** (`str`, optional): 
  - Description: An optional prompt to guide the transcription process.
  - Default: `None`
  - Value Range: Any string that serves as a prompt for the transcription.

- **audio_file** (`str`): 
  - Description: The path to the audio file to be transcribed. This can be a local file path or a URL.
  - Value Range: Must be a valid file path or URL.

- **device** (`str`, optional): 
  - Description: The device to use for computation (e.g., 'cuda' for GPU or 'cpu' for CPU).
  - Default: `'cuda'`
  - Value Range: Must be either 'cuda' or 'cpu'.

- **compute_type** (`str`, optional): 
  - Description: The type of computation to perform (e.g., 'float16' for half-precision).
  - Default: `'float16'`
  - Value Range: Must be a valid compute type supported by the Whisper library.

### Return Value
- **None**: The function does not return a value. Instead, it saves the transcribed text as an SRT file in the output directory.

### Purpose
The `speech_to_text` function is designed to convert spoken language in audio files into written text, facilitating the creation of subtitles or transcripts for various applications. It leverages the capabilities of the Whisper model to provide accurate transcriptions in multiple languages. 

### Example Usage
```python
speech_to_text(audio_file='path/to/audio/file.wav', language='en')
```

### Error Handling
- Raises an exception if the specified `audio_file` does not exist or is not accessible.
- The function will also raise an exception if the model name is invalid or if there are issues with the audio processing.

