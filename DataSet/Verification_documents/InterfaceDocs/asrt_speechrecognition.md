# API Documentation for ModelSpeech Class

## Class: ModelSpeech
The `ModelSpeech` class is designed for speech recognition using an N-Gram language model specifically tailored for ASRT (Automatic Speech Recognition Tool). It integrates an acoustic model and speech features to facilitate the training and evaluation of speech recognition models.

### Attributes:
- **speech_model**: An instance of the acoustic model (BaseModel class).
- **speech_features**: An instance of the speech feature extractor (SpeechFeatureMeta class).
- **max_label_length**: An integer representing the maximum length of labels (default is 64).
- **trained_model**: The trained Keras model instance.
- **base_model**: The base model instance from the speech model.

### Method: `__init__(self, speech_model, speech_features, max_label_length=64)`
#### Parameters:
- **speech_model**: An instance of the acoustic model (BaseModel class).
- **speech_features**: An instance of the speech feature extractor (SpeechFeatureMeta class).
- **max_label_length**: (Optional) An integer specifying the maximum length of labels. Default is 64.

#### Purpose:
Initializes the `ModelSpeech` class with the specified acoustic model and speech features, setting up the necessary attributes for training and evaluation.

---

### Method: `train_model(self, optimizer, data_loader, epochs=1, save_step=1, batch_size=16, last_epoch=0, call_back=None)`
#### Parameters:
- **optimizer**: An instance of a TensorFlow Keras optimizer.
- **data_loader**: An instance of the data loader (SpeechData class).
- **epochs**: (Optional) An integer specifying the number of epochs to train. Default is 1.
- **save_step**: (Optional) An integer specifying how often to save the model (in epochs). Default is 1.
- **batch_size**: (Optional) An integer specifying the size of mini-batches. Default is 16.
- **last_epoch**: (Optional) An integer indicating the last completed epoch for resuming training. Default is 0.
- **call_back**: (Optional) A Keras callback function for additional functionality during training.

#### Purpose:
Trains the speech recognition model using the specified optimizer and data loader, iterating through the specified number of epochs and saving the model at defined intervals.

---

### Method: `load_model(self, filename)`
#### Parameters:
- **filename**: A string representing the path to the model weights file.

#### Purpose:
Loads the model parameters from the specified file.

---

### Method: `save_model(self, filename)`
#### Parameters:
- **filename**: A string representing the path where the model weights will be saved.

#### Purpose:
Saves the current model parameters to the specified file.

---

### Method: `evaluate_model(self, data_loader, data_count=-1, out_report=False, show_ratio=True, show_per_step=100)`
#### Parameters:
- **data_loader**: An instance of the data loader (SpeechData class).
- **data_count**: (Optional) An integer specifying the number of samples to evaluate. Default is -1, which means all available data will be used.
- **out_report**: (Optional) A boolean indicating whether to output a report file. Default is False.
- **show_ratio**: (Optional) A boolean indicating whether to show progress ratio during evaluation. Default is True.
- **show_per_step**: (Optional) An integer specifying how often to show progress (in steps). Default is 100.

#### Purpose:
Evaluates the model's performance on the provided data, calculating the word error ratio and optionally generating a report.

---

### Method: `predict(self, data_input)`
#### Parameters:
- **data_input**: A NumPy array containing the input data for prediction.

#### Returns:
- A list of predicted labels corresponding to the input data.

#### Purpose:
Generates predictions for the given input data using the trained model.

---

### Method: `recognize_speech(self, wavsignal, fs)`
#### Parameters:
- **wavsignal**: A NumPy array representing the audio signal to be recognized.
- **fs**: An integer representing the sample rate of the audio signal.

#### Returns:
- A list of recognized symbols (e.g., phonemes or words) from the audio signal.

#### Purpose:
Performs speech recognition on the provided audio signal, returning the recognized text.

---

### Method: `recognize_speech_from_file(self, filename)`
#### Parameters:
- **filename**: A string representing the path to the audio file to be recognized.

#### Returns:
- A list of recognized symbols (e.g., phonemes or words) from the audio file.

#### Purpose:
Loads an audio file and performs speech recognition, returning the recognized text.

---

### Property: `model`
#### Returns:
- The trained Keras model instance.

#### Purpose:
Provides access to the underlying TensorFlow Keras model for further manipulation or evaluation.

# API Documentation for ModelLanguage Class

## Class: ModelLanguage
The `ModelLanguage` class implements an N-Gram language model specifically designed for the ASRT (Automatic Speech Recognition Tool). It provides functionality to load language models and convert pinyin input into text.

### Attributes:
- **model_path**: A string representing the path to the directory containing the language model files.
- **dict_pinyin**: A dictionary that maps pinyin to corresponding Chinese characters.
- **model1**: A dictionary representing the unigram language model.
- **model2**: A dictionary representing the bigram language model.

### Method: `__init__(self, model_path: str)`
#### Parameters:
- **model_path**: A string specifying the path to the directory where the language model files are stored.

#### Purpose:
Initializes the `ModelLanguage` class with the specified model path and sets up the necessary attributes for loading and processing the language model.

---

### Method: `load_model(self)`
#### Parameters:
- None

#### Returns:
- A tuple containing:
  - **dict_pinyin**: A dictionary mapping pinyin to corresponding Chinese characters.
  - **model1**: A dictionary representing the unigram language model.
  - **model2**: A dictionary representing the bigram language model.

#### Purpose:
Loads the N-Gram language model into memory by reading the necessary files and populating the internal dictionaries.

---

### Method: `pinyin_to_text(self, list_pinyin: list, beam_size: int = 100) -> str`
#### Parameters:
- **list_pinyin**: A list of strings representing pinyin input.
- **beam_size**: (Optional) An integer specifying the number of top results to consider during decoding. Default is 100.

#### Returns:
- A string representing the converted text from the provided pinyin input.

#### Purpose:
Converts a list of pinyin strings into corresponding Chinese text, utilizing a beam search approach to optimize the results.

---

### Method: `pinyin_stream_decode(self, temple_result: list, item_pinyin: str, beam_size: int = 100) -> list`
#### Parameters:
- **temple_result**: A list of intermediate results from previous decoding steps.
- **item_pinyin**: A string representing the current pinyin input to decode.
- **beam_size**: (Optional) An integer specifying the number of top results to return. Default is 100.

#### Returns:
- A list of lists, where each inner list contains a possible sequence of characters and its associated probability.

#### Purpose:
Performs a stream decoding of the provided pinyin input, returning intermediate results for each character based on the current state and the language model. This method allows for incremental decoding of pinyin into text.

--- 

### Example Usage:
```python
if __name__ == '__main__':
    ml = ModelLanguage('model_language')
    ml.load_model()

    _str_pinyin = ['zhe4', 'zhen1', 'shi4', 'ji2', 'hao3', 'de5']
    _RESULT = ml.pinyin_to_text(_str_pinyin)
    print('语音转文字结果:\n', _RESULT)
```

This example demonstrates how to create an instance of the `ModelLanguage` class, load the language model, and convert a list of pinyin strings into text.

