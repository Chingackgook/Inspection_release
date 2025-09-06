# API Documentation

## Function: `generate`

### Description
The `generate` function utilizes a language model to produce high-quality, random examples that conform to a specified type or description. The generated examples are returned as a list.

### Parameters

- **target** (`TargetType[T] | None`): 
  - Description: The type of entities to generate. This parameter specifies the expected type of the generated examples.
  - Value Range: Can be any valid type or `None`.

- **n** (`int`): 
  - Description: The number of examples to generate.
  - Value Range: Must be a positive integer. Defaults to `1`.

- **instructions** (`str | None`): 
  - Description: Optional instructions that provide specific guidance on what kinds of examples to create.
  - Value Range: Can be any string or `None`.

- **agent** (`Agent | None`): 
  - Description: An optional custom agent to use for generation. If not provided, the default agent will be used.
  - Value Range: Can be an instance of `Agent` or `None`.

- **thread** (`Thread | str | None`): 
  - Description: An optional thread for maintaining conversation context. This can be either a `Thread` object or a string representing the thread ID.
  - Value Range: Can be an instance of `Thread`, a string, or `None`.

- **context** (`dict[str, Any] | None`): 
  - Description: An optional dictionary of additional context to include in the task.
  - Value Range: Can be any dictionary or `None`.

- **handlers** (`list[Handler | AsyncHandler] | None`): 
  - Description: An optional list of handlers to use for the task.
  - Value Range: Can be a list containing instances of `Handler` or `AsyncHandler`, or `None`.

- **prompt** (`str | None`): 
  - Description: An optional prompt to use for the task. If not provided, the default prompt will be used.
  - Value Range: Can be any string or `None`.

### Returns
- **list[T]**: A list containing `n` generated entities of type `T`. The entities are generated based on the specified parameters.

### Example
```python
examples = generate(target=MyType, n=5, instructions="Generate random examples of MyType.")
```

### Purpose
The `generate` function is designed to facilitate the creation of random examples that adhere to a specified type or description, making it useful for testing, data augmentation, or any scenario where example data is needed.

# API Documentation

## Function: `run`

### Description
The `run` function executes a set of instructions, potentially utilizing various tools and agents, and returns a result of a specified type. It is designed to facilitate the execution of tasks in an asynchronous environment while providing options for error handling and context management.

### Parameters

- **instructions** (`str | Sequence[UserContent]`): 
  - Description: The instructions to be executed. This can be a single string or a sequence of user content messages.
  - Value Range: Can be a string or a sequence of `UserContent` instances.

- **result_type** (`type[T]`): 
  - Description: The expected type of the result returned by the function.
  - Value Range: Any valid Python type. Defaults to `str`.

- **tools** (`list[Callable[..., Any]]`): 
  - Description: A list of callable tools that can be used during the execution of the instructions.
  - Value Range: Can be an empty list or a list of callable functions.

- **thread** (`Thread | str | None`): 
  - Description: An optional thread for maintaining conversation context. This can be either a `Thread` object or a string representing the thread ID.
  - Value Range: Can be an instance of `Thread`, a string, or `None`.

- **agents** (`list[Actor] | None`): 
  - Description: An optional list of agents to use during the execution. If not provided, the default agents will be used.
  - Value Range: Can be a list of `Actor` instances or `None`.

- **raise_on_failure** (`bool`): 
  - Description: A flag indicating whether to raise an exception if the execution fails.
  - Value Range: Must be a boolean value. Defaults to `True`.

- **handlers** (`list[Handler | AsyncHandler] | None`): 
  - Description: An optional list of handlers to use for the task.
  - Value Range: Can be a list containing instances of `Handler` or `AsyncHandler`, or `None`.

- **kwargs** (`Any`): 
  - Description: Additional keyword arguments that may be passed to the underlying asynchronous function.
  - Value Range: Any additional keyword arguments.

### Returns
- **T**: The result of the execution, which is of the type specified by `result_type`. The return value will depend on the instructions executed and the context provided.

### Example
```python
result = run(instructions="Calculate the sum of 1 and 2.", result_type=int)
```

### Purpose
The `run` function is designed to streamline the execution of tasks that require instructions, tools, and agents, while also providing flexibility in handling results and errors. It is particularly useful in scenarios where asynchronous execution is needed, such as in chatbots, automated workflows, or interactive applications.

