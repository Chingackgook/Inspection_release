# API Documentation

## Class: Sequential

### Description
The `Sequential` class applies a linear chain of modules or callables. It is designed for simple architectures where the output of one module is the input to the next. This class is part of the Sonnet library and is useful for constructing feedforward neural networks.

### Attributes
- **layers**: A list of callables (modules) that will be executed in sequence. Each callable should accept the output of the previous callable as its input.

### Methods

#### `__init__(self, layers: Optional[Iterable[Callable[..., Any]]] = None, name: Optional[str] = None)`

- **Description**: Initializes a `Sequential` object with a list of layers.
- **Parameters**:
  - `layers` (Optional[Iterable[Callable[..., Any]]]): A collection of callables (modules) to be executed in sequence. If `None`, an empty list is initialized. 
    - **Value Range**: Can be any iterable of callables, including functions or instances of classes that implement a `__call__` method.
  - `name` (Optional[str]): An optional name for the module.
    - **Value Range**: Any string or `None`.
- **Return Value**: None
- **Purpose**: To create a sequential model by specifying the layers that will be executed in order.

#### `__call__(self, inputs, *args, **kwargs)`

- **Description**: Executes the sequential layers on the provided inputs.
- **Parameters**:
  - `inputs`: The input data to be processed through the sequential layers.
    - **Value Range**: Can be any tensor-like structure compatible with the first layer's input requirements.
  - `*args`: Additional positional arguments to be passed to the first layer.
    - **Value Range**: Any additional arguments that the first layer may require.
  - `**kwargs`: Additional keyword arguments to be passed to the first layer.
    - **Value Range**: Any additional keyword arguments that the first layer may require.
- **Return Value**: The output of the last layer after processing the inputs through all layers in the sequence.
- **Purpose**: To apply the sequence of layers to the input data, returning the final output after all transformations.

### Example Usage
```python
mlp = Sequential([
    snt.Linear(1024),
    tf.nn.relu,
    snt.Linear(10),
])
output = mlp(tf.random.normal([8, 100]))
```

### Notes
- The `Sequential` class is limited to architectures where each layer's input is the output of the previous layer. For more complex architectures, consider subclassing `snt.Module` and implementing a custom `__call__` method.

