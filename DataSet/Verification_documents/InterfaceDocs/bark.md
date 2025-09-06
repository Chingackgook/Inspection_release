# API Documentation

## Function: `generate_audio`

### Description
Generates an audio array from the provided input text. This function converts text into a semantic representation and then into an audio waveform.

### Parameters
- **text** (str): The text to be converted into audio.
- **history_prompt** (Optional[Union[Dict, str]]): A history choice for audio cloning. Default is `None`.
- **text_temp** (float): The generation temperature for text (1.0 for more diverse output, 0.0 for more conservative). Default is `0.7`.
- **waveform_temp** (float): The generation temperature for waveform (1.0 for more diverse output, 0.0 for more conservative). Default is `0.7`.
- **silent** (bool): If `True`, disables the progress bar. Default is `False`.
- **output_full** (bool): If `True`, returns the full generation to be used as a history prompt. Default is `False`.

### Returns
- **numpy.ndarray**: An audio array sampled at 24 kHz. If `output_full` is `True`, it returns a tuple containing the full generation and the audio array.

### Purpose
This function is designed to convert textual input into audio format, allowing for the generation of speech or sound from written content.

---

## Function: `text_to_semantic`

### Description
Generates a semantic array from the provided text. This function prepares the text for audio generation by converting it into a semantic representation.

### Parameters
- **text** (str): The text to be converted into a semantic array.
- **history_prompt** (Optional[Union[Dict, str]]): A history choice for audio cloning. Default is `None`.
- **temp** (float): The generation temperature (1.0 for more diverse output, 0.0 for more conservative). Default is `0.7`.
- **silent** (bool): If `True`, disables the progress bar. Default is `False`.

### Returns
- **numpy.ndarray**: A semantic array that can be fed into the `semantic_to_waveform` function.

### Purpose
This function serves to transform text into a semantic format, which is a necessary step before generating audio from the text.

---

## Function: `semantic_to_waveform`

### Description
Generates an audio array from the provided semantic input. This function takes the semantic representation and converts it into an audio waveform.

### Parameters
- **semantic_tokens** (np.ndarray): The semantic token output from the `text_to_semantic` function.
- **history_prompt** (Optional[Union[Dict, str]]): A history choice for audio cloning. Default is `None`.
- **temp** (float): The generation temperature (1.0 for more diverse output, 0.0 for more conservative). Default is `0.7`.
- **silent** (bool): If `True`, disables the progress bar. Default is `False`.
- **output_full** (bool): If `True`, returns the full generation to be used as a history prompt. Default is `False`.

### Returns
- **numpy.ndarray**: An audio array sampled at 24 kHz. If `output_full` is `True`, it returns a tuple containing the full generation and the audio array.

### Purpose
This function is responsible for converting semantic tokens into an audio waveform, completing the process of generating audio from text.

---

## Function: `save_as_prompt`

### Description
Saves the full generation of audio prompts into a specified file format. This function allows for the storage of semantic, coarse, and fine prompts for future use.

### Parameters
- **filepath** (str): The path where the full generation will be saved. Must end with ".npz".
- **full_generation** (dict): A dictionary containing the full generation data, which must include "semantic_prompt", "coarse_prompt", and "fine_prompt".

### Returns
- **None**: This function does not return a value.

### Purpose
This function is used to save the generated audio prompts into a file, enabling users to store and retrieve audio generation data as needed.