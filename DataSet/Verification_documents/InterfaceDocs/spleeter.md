# API Documentation

## Function: `create_estimator`

### Description
Initializes a TensorFlow estimator that will perform source separation.

### Parameters
- **params (Dict)**: A dictionary of parameters for building the model.
- **MWF (bool)**: A flag indicating whether the Wiener filter should be enabled. Accepts `True` or `False`.

### Returns
- **tf.Tensor**: A TensorFlow estimator configured for source separation.

### Purpose
This function sets up the TensorFlow estimator required for performing audio source separation using the specified model parameters.

---

## Class: `Separator`

### Description
A wrapper class for performing audio source separation using deep learning models.

### Attributes
- **_params (Dict)**: Configuration parameters for the model.
- **_sample_rate (int)**: Sample rate for audio processing.
- **_MWF (bool)**: Indicates if the Wiener filter is enabled.
- **_tf_graph (tf.Graph)**: TensorFlow computation graph.
- **_prediction_generator (Optional[Generator])**: Generator for predictions from the estimator.
- **_input_provider (Optional[Any])**: Input provider for audio data.
- **_builder (Optional[Any])**: Estimator specification builder.
- **_features (Optional[Any])**: Features for the model input.
- **_session (Optional[tf.Session])**: TensorFlow session for running the model.
- **_pool (Optional[Pool])**: Pool for multiprocessing tasks.
- **_tasks (List)**: List of tasks for asynchronous processing.
- **estimator (Optional[tf.estimator.Estimator])**: TensorFlow estimator for predictions.

### Method: `__init__`

#### Description
Initializes the `Separator` class with specified parameters.

#### Parameters
- **params_descriptor (str)**: Descriptor for TensorFlow parameters to be used.
- **MWF (bool)**: (Optional) `True` if the Wiener filter should be used, `False` otherwise. Default is `False`.
- **multiprocess (bool)**: (Optional) Enable multi-processing. Default is `True`.

#### Returns
- **None**

#### Purpose
This constructor initializes the `Separator` instance with the necessary configurations for audio source separation.

---

### Method: `join`

#### Description
Waits for all pending tasks to be finished.

#### Parameters
- **timeout (int)**: (Optional) Task waiting timeout in seconds. Default is `200`.

#### Returns
- **None**

#### Purpose
This method ensures that all asynchronous tasks are completed before proceeding, allowing for controlled execution flow.

---

### Method: `separate`

#### Description
Performs source separation on a given waveform.

#### Parameters
- **waveform (np.ndarray)**: Waveform to be separated, represented as a NumPy array.
- **audio_descriptor (Optional[str])**: (Optional) String describing the waveform (e.g., filename). Default is an empty string.

#### Returns
- **Dict**: A dictionary containing separated waveforms, where keys are instrument names and values are their corresponding waveforms.

#### Purpose
This method takes an audio waveform and separates it into distinct sources (e.g., vocals, accompaniment) using the configured model.

---

### Method: `separate_to_file`

#### Description
Performs source separation and exports the results to files using the specified audio adapter.

#### Parameters
- **audio_descriptor (AudioDescriptor)**: Descriptor for the audio to be separated, used by the audio adapter to load audio data.
- **destination (str)**: Target directory to write output files.
- **audio_adapter (Optional[AudioAdapter])**: (Optional) Audio adapter to use for input/output operations. Default is `None`.
- **offset (float)**: (Optional) Offset in seconds for loading the audio. Default is `0`.
- **duration (float)**: (Optional) Duration in seconds for loading the audio. Default is `600.0`.
- **codec (Codec)**: (Optional) Export codec. Default is `Codec.WAV`.
- **bitrate (str)**: (Optional) Export bitrate. Default is `"128k"`.
- **filename_format (str)**: (Optional) Format for generated filenames. Default is `"{filename}/{instrument}.{codec}"`.
- **synchronous (bool)**: (Optional) If `True`, the operation is synchronous. Default is `True`.

#### Returns
- **None**

#### Purpose
This method separates the audio sources and saves them to specified files in the given directory, allowing for easy access to the separated tracks.

---

### Method: `save_to_file`

#### Description
Exports a dictionary of separated sources to files.

#### Parameters
- **sources (Dict)**: Dictionary of sources to be exported, where keys are instrument names and values are `N x 2` NumPy arrays containing the corresponding instrument waveforms.
- **audio_descriptor (AudioDescriptor)**: Descriptor for the audio to be separated, used by the audio adapter to load audio data.
- **destination (str)**: Target directory to write output files.
- **filename_format (str)**: (Optional) Format for generated filenames. Default is `"{filename}/{instrument}.{codec}"`.
- **codec (Codec)**: (Optional) Export codec. Default is `Codec.WAV`.
- **audio_adapter (Optional[AudioAdapter])**: (Optional) Audio adapter to use for input/output operations. Default is `None`.
- **bitrate (str)**: (Optional) Export bitrate. Default is `"128k"`.
- **synchronous (bool)**: (Optional) If `True`, the operation is synchronous. Default is `True`.

#### Returns
- **None**

#### Purpose
This method saves the separated audio sources to files, allowing users to access and utilize the separated tracks in their desired format.