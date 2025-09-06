# API Documentation

## Functions

### `get_predictor() -> Model`
- **Description**: This function retrieves the OpenNSFW model predictor. It initializes the model if it has not been created yet, ensuring thread safety with a lock.
- **Parameters**: None
- **Return Value**: Returns an instance of the OpenNSFW model (`Model`).
- **Usage**: This function is used to obtain the model for making predictions on images or video frames.

---

### `clear_predictor() -> None`
- **Description**: This function clears the current OpenNSFW model predictor, allowing it to be re-initialized on the next call to `get_predictor()`.
- **Parameters**: None
- **Return Value**: None
- **Usage**: This function is useful for freeing up resources or resetting the model state.

---

### `predict_frame(target_frame: Frame) -> bool`
- **Parameters**:
  - `target_frame` (Frame): A frame of video or an image represented as a numpy array.
    - **Value Range**: The input should be a valid image array compatible with the OpenNSFW model.
- **Return Value**: Returns a boolean value (`True` or `False`).
- **Description**: This function predicts whether the given frame contains adult content based on the OpenNSFW model. It preprocesses the image and checks if the probability of adult content exceeds the defined threshold (`MAX_PROBABILITY`).
- **Usage**: This function is used to analyze individual frames from videos or images for adult content.

---

### `predict_image(target_path: str) -> bool`
- **Parameters**:
  - `target_path` (str): The file path to the image that needs to be analyzed.
    - **Value Range**: The path should point to a valid image file supported by the OpenNSFW model.
- **Return Value**: Returns a boolean value (`True` or `False`).
- **Description**: This function predicts whether the image at the specified path contains adult content. It uses the OpenNSFW model to evaluate the image and returns `True` if the probability exceeds the defined threshold (`MAX_PROBABILITY`).
- **Usage**: This function is used to analyze static images for adult content.

---

### `predict_video(target_path: str) -> bool`
- **Parameters**:
  - `target_path` (str): The file path to the video that needs to be analyzed.
    - **Value Range**: The path should point to a valid video file supported by the OpenNSFW model.
- **Return Value**: Returns a boolean value (`True` or `False`).
- **Description**: This function predicts whether the video at the specified path contains adult content by analyzing its frames. It checks if any frame's probability of adult content exceeds the defined threshold (`MAX_PROBABILITY`).
- **Usage**: This function is used to analyze videos for adult content by evaluating multiple frames.

--- 

This documentation provides a clear understanding of the purpose, parameters, and return values of each function in the `predict_image_predict_video` API implementation.

