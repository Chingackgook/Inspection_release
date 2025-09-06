# API Documentation

## Function: `load_cpk`

### Description
Loads a checkpoint for a model and optionally an optimizer from a specified path.

### Parameters
- **checkpoint_path** (str): The path to the checkpoint file.
- **model** (torch.nn.Module, optional): The model to load the state dictionary into. Default is None.
- **optimizer** (torch.optim.Optimizer, optional): The optimizer to load the state dictionary into. Default is None.
- **device** (str): The device to load the model onto (e.g., "cpu", "cuda"). Default is "cpu".

### Returns
- **int**: The epoch number from the loaded checkpoint.

### Purpose
This function is used to restore the state of a model and optimizer from a saved checkpoint, allowing for resuming training or inference from a specific point.

---

## Class: `Audio2Coeff`

### Description
A class that maps audio features to facial expression coefficients, enabling synchronized audio-visual performance through cross-modal understanding.

### Attributes
- **audio2pose_model** (Audio2Pose): The model responsible for generating pose coefficients from audio input.
- **audio2exp_model** (Audio2Exp): The model responsible for generating expression coefficients from audio input.
- **device** (str): The device on which the models are loaded (e.g., "cpu", "cuda").

### Method: `__init__`

#### Description
Initializes the `Audio2Coeff` class by loading the necessary models and their configurations.

#### Parameters
- **sadtalker_path** (dict): A dictionary containing paths to configuration and checkpoint files.
  - **audio2pose_yaml_path** (str): Path to the YAML configuration file for the audio2pose model.
  - **audio2exp_yaml_path** (str): Path to the YAML configuration file for the audio2exp model.
  - **checkpoint** (str): Path to the checkpoint file (if using safetensors).
  - **audio2pose_checkpoint** (str): Path to the audio2pose checkpoint file.
  - **audio2exp_checkpoint** (str): Path to the audio2exp checkpoint file.
  - **use_safetensor** (bool): Flag indicating whether to use safetensors for loading checkpoints.
- **device** (str): The device to load the models onto (e.g., "cpu", "cuda").

#### Purpose
This method sets up the `Audio2Coeff` instance by loading the configurations and models required for audio-to-coefficient mapping.

---

### Method: `generate`

#### Description
Generates facial expression and pose coefficients from a given audio input batch.

#### Parameters
- **batch** (dict): A dictionary containing input data for the model.
  - **pic_name** (str): The name of the picture associated with the audio.
  - **audio_name** (str): The name of the audio file.
  - **class** (torch.LongTensor): The class ID for the pose style.
- **coeff_save_dir** (str): The directory where the generated coefficients will be saved.
- **pose_style** (int): The ID of the pose style to be used (should be in the range of available styles).
- **ref_pose_coeff_path** (str, optional): Path to a reference pose coefficient file. Default is None.

#### Returns
- **str**: The file path of the saved coefficients in .mat format.

#### Purpose
This method processes the input audio batch to generate and save the corresponding facial expression and pose coefficients.

---

### Method: `using_refpose`

#### Description
Adjusts the predicted coefficients using a reference pose coefficient file.

#### Parameters
- **coeffs_pred_numpy** (numpy.ndarray): The predicted coefficients to be adjusted.
- **ref_pose_coeff_path** (str): Path to the reference pose coefficient file.

#### Returns
- **numpy.ndarray**: The adjusted coefficients after applying the reference pose.

#### Purpose
This method modifies the predicted coefficients by incorporating information from a reference pose, allowing for more accurate facial expressions and poses based on a given reference.