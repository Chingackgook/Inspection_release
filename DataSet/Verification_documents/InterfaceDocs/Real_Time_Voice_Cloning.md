# API Documentation

## Class: `Synthesizer`

The `Synthesizer` class is responsible for converting text and speaker voice features into mel spectrograms for speech synthesis. It also provides audio preprocessing and postprocessing functionalities.

### Attributes:
- **sample_rate**: (int) The sample rate used for audio processing, defined in the hyperparameters.
- **hparams**: (object) Hyperparameters used for configuring the synthesizer model.
- **model_fpath**: (Path) The file path to the trained model weights.
- **verbose**: (bool) If set to `True`, the synthesizer will print additional information during operation.
- **device**: (torch.device) The device on which the model will run (either CPU or GPU).
- **_model**: (Tacotron) The Tacotron model instance, which is initialized when the model is loaded.

### Method: `__init__(self, model_fpath: Path, verbose=True)`

#### Parameters:
- **model_fpath**: (Path) The path to the trained model file.
- **verbose**: (bool, optional) If `False`, prints less information when using the model. Default is `True`.

#### Returns:
- None

#### Description:
Initializes the `Synthesizer` instance, setting up the model file path and verbosity level. It checks for GPU availability and sets the device accordingly. The model is not loaded into memory until the `load()` method is called.

---

### Method: `is_loaded(self)`

#### Parameters:
- None

#### Returns:
- (bool) Returns `True` if the model is loaded in memory, otherwise `False`.

#### Description:
Checks whether the synthesizer model is currently loaded in memory.

---

### Method: `load(self)`

#### Parameters:
- None

#### Returns:
- None

#### Description:
Instantiates and loads the Tacotron model using the weights file specified during initialization. The model is set to evaluation mode after loading. If verbosity is enabled, it prints the name of the loaded synthesizer and the training step.

---

### Method: `synthesize_spectrograms(self, texts: List[str], embeddings: Union[np.ndarray, List[np.ndarray]], return_alignments=False)`

#### Parameters:
- **texts**: (List[str]) A list of N text prompts to be synthesized.
- **embeddings**: (Union[np.ndarray, List[np.ndarray]]) A numpy array or list of speaker embeddings of shape (N, 256).
- **return_alignments**: (bool, optional) If `True`, returns a matrix representing the alignments between the characters and each decoder output step for each spectrogram. Default is `False`.

#### Returns:
- (List[np.ndarray]) A list of N mel spectrograms as numpy arrays of shape (80, Mi), where Mi is the sequence length of spectrogram i. If `return_alignments` is `True`, it also returns the alignments.

#### Description:
Synthesizes mel spectrograms from the provided text prompts and speaker embeddings. The method preprocesses the text inputs, batches them for processing, and generates the spectrograms using the loaded model. It trims silence from the end of each spectrogram before returning the results. If verbosity is enabled, it prints progress information during synthesis.

