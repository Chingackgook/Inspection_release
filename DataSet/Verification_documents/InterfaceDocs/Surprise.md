# API Documentation

## Class: KNNBasic

### `__init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs)`
Initializes the KNNBasic algorithm.

**Parameters:**
- `k` (int, optional): The maximum number of neighbors to consider. Default is `40`.
- `min_k` (int, optional): The minimum number of neighbors required. Default is `1`.
- `sim_options` (dict, optional): Options for the similarity measure.
- `verbose` (bool, optional): Whether to print trace messages. Default is `True`.
- `**kwargs`: Additional keyword arguments.

**Attributes:**
- `k`: Maximum number of neighbors.
- `min_k`: Minimum number of neighbors.
- `verbose`: Stores verbosity option.
- `bsl_options`: Stores baseline options.
- `sim_options`: Stores similarity options.
- `trainset`: The training dataset.

### `fit(self, trainset)`
Trains the KNNBasic algorithm on a given training set.

**Parameters:**
- `trainset`: A training set returned by the dataset's folds method.

**Returns:**
- `self`: The instance of the class.

**Purpose:**
Initializes the similarity matrix after fitting the model and sets the training dataset.

### `predict(self, uid, iid, r_ui=None, clip=True, verbose=False)`
Computes the rating prediction for a given user and item.

**Parameters:**
- `uid`: (Raw) id of the user.
- `iid`: (Raw) id of the item.
- `r_ui` (float, optional): The true rating. Default is `None`.
- `clip` (bool, optional): Whether to clip the estimation into the rating scale. Default is `True`.
- `verbose` (bool, optional): Whether to print details of the prediction. Default is `False`.

**Returns:**
- `Prediction`: A prediction object containing user id, item id, true rating, estimated rating, and additional details.

**Purpose:**
Converts raw ids to inner ids and estimates the rating.

### `default_prediction(self)`
Returns the default prediction when prediction is impossible.

**Returns:**
- `float`: The mean of all ratings in the training set.

**Purpose:**
Provides a fallback prediction when user/item is unknown.

### `test(self, testset, verbose=False)`
Tests the algorithm on a given test set.

**Parameters:**
- `testset`: A test set returned by a cross-validation iterator or the build_testset method.
- `verbose` (bool, optional): Whether to print details for each prediction. Default is `False`.

**Returns:**
- `list`: A list of prediction objects containing all estimated ratings.

**Purpose:**
Estimates all ratings in the given test set.

### `compute_baselines(self)`
Computes user and item baselines.

**Returns:**
- `tuple`: A tuple containing user and item baselines.

**Purpose:**
Calculates baselines based on the specified method in `bsl_options`.

### `compute_similarities(self)`
Builds the similarity matrix.

**Returns:**
- `array`: The similarity matrix.

**Purpose:**
Computes the similarity matrix based on the specified method in `sim_options`.

### `get_neighbors(self, iid, k)`
Returns the `k` nearest neighbors of `iid`.

**Parameters:**
- `iid` (int): The inner id of the user or item.
- `k` (int): The number of neighbors to retrieve.

**Returns:**
- `list`: A list of the `k` inner ids of the closest users or items.

**Purpose:**
Finds the nearest neighbors based on the similarity measure.

### `switch(self, u_stuff, i_stuff)`
Returns `u_stuff` and `i_stuff` based on the `user_based` field.

**Parameters:**
- `u_stuff`: User-related data.
- `i_stuff`: Item-related data.

**Returns:**
- `tuple`: A tuple containing user and item data based on the `user_based` field.

**Purpose:**
Facilitates switching between user and item data based on the algorithm's configuration.

### `estimate(self, u, i)`
Estimates the rating for a given user and item.

**Parameters:**
- `u`: Inner id of the user.
- `i`: Inner id of the item.

**Returns:**
- `float`: The estimated rating.
- `dict`: A dictionary containing additional details about the estimation.

**Raises:**
- `PredictionImpossible`: If the user or item is unknown or if there are not enough neighbors.

**Purpose:**
Calculates the predicted rating based on the nearest neighbors' ratings.

