# API Documentation

## Class: TextClassifier

### `__init__`
Initializes the TextClassifier with the specified parameters.

#### Parameters:
- `num_classes` (Optional[int], optional): Number of classes. If None, it will be inferred from the data.
- `multi_label` (bool, optional): Indicates if the classification is multi-label. Defaults to False.
- `loss` (types.LossType, optional): A Keras loss function. Defaults to 'binary_crossentropy' or 'categorical_crossentropy' based on the number of classes.
- `metrics` (Optional[types.MetricsType], optional): A list of Keras metrics. Defaults to 'accuracy'.
- `project_name` (str, optional): The name of the AutoModel. Defaults to 'text_classifier'.
- `max_trials` (int, optional): The maximum number of different Keras Models to try. Defaults to 100.
- `directory` (Union[str, Path, None], optional): The path to a directory for storing the search outputs. Defaults to None.
- `objective` (str, optional): Name of model metric to minimize or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
- `tuner` (Union[str, Type[tuner.AutoTuner]], optional): Tuner type or instance. Defaults to task-specific tuner.
- `overwrite` (bool, optional): If `False`, reloads an existing project of the same name if one is found. Defaults to `False`.
- `seed` (Optional[int], optional): Random seed. Defaults to None.
- `max_model_size` (Optional[int], optional): Maximum number of scalars in the parameters of a model. Models larger than this are rejected. Defaults to None.
- `**kwargs`: Additional keyword arguments.

### `fit`
Searches for the best model and hyperparameters for the AutoModel.

#### Parameters:
- `x`: numpy.ndarray or tensorflow.Dataset. Training data x. The input data should be one-dimensional strings.
- `y`: numpy.ndarray or tensorflow.Dataset. Training data y. It can be raw labels, one-hot encoded, or binary encoded.
- `epochs` (int, optional): The number of epochs to train each model during the search. Defaults to a maximum of 1000 epochs.
- `callbacks` (list, optional): List of Keras callbacks to apply during training and validation.
- `validation_split` (float, optional): Fraction of the training data to be used as validation data. Defaults to 0.2.
- `validation_data`: Data on which to evaluate the loss and any model metrics at the end of each epoch. Overrides `validation_split`.
- `**kwargs`: Any arguments supported by `keras.Model.fit`.

#### Returns:
- `history`: A Keras History object corresponding to the best model, containing training and validation loss and metrics values.

### `predict`
Predicts the output for a given testing data.

#### Parameters:
- `x`: Any allowed types according to the input node. Testing data.
- `batch_size` (int, optional): Number of samples per batch. Defaults to 32.
- `verbose` (int, optional): Verbosity mode. 0 = silent, 1 = progress bar. Defaults to 1.
- `**kwargs`: Any arguments supported by `keras.Model.predict`.

#### Returns:
- A list of numpy.ndarray objects or a single numpy.ndarray containing the predicted results.

### `evaluate`
Evaluates the best model for the given data.

#### Parameters:
- `x`: Any allowed types according to the input node. Testing data.
- `y`: Any allowed types according to the head. Testing targets. Defaults to None.
- `batch_size` (int, optional): Number of samples per batch. Defaults to 32.
- `verbose` (int, optional): Verbosity mode. 0 = silent, 1 = progress bar. Defaults to 1.
- `**kwargs`: Any arguments supported by `keras.Model.evaluate`.

#### Returns:
- Scalar test loss (if the model has a single output and no metrics) or list of scalars (if the model has multiple outputs and/or metrics).

