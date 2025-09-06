Based on the provided API documentation, here is the classification of the functions and methods:

### Top-Level Functions
The following are top-level functions:
1. `available_models()`
2. `load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None)`
3. `tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False)`

### Class Methods
There are no class methods listed in the provided documentation. All functions appear to be top-level functions with no indication of being part of a class.

### Total Number of Interface Classes
The documentation does not mention any interface classes. Therefore, the total number of interface classes is **0**.

Let's address your questions one by one:

### Ques 1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
In the provided interface documentation, there are no class methods or classes mentioned that would require initialization. All functions are top-level functions, meaning you do not need to create instances of any classes in `create_interface_objects`. Therefore, initialization is unnecessary.

### Ques 2: Which top-level functions should be mapped to `run`?
The top-level functions from the interface documentation that should be mapped to `run` are:
1. `available_models()`
2. `load(name, device, jit, download_root)`
3. `tokenize(texts, context_length, truncate)`

These functions can be called directly in the `run` method, and you can use the `dispatch_key` parameter to determine which function to execute.

### Ques 3: Which class methods should be mapped to `run`?
As per the provided interface documentation, there are no class methods mentioned. All functions are top-level functions, and there are no classes that you need to consider for method mapping. Therefore, there are no class methods to be mapped to `run`.

In summary:
- No initialization of class objects is necessary in `create_interface_objects`.
- The top-level functions `available_models`, `load`, and `tokenize` should be mapped to `run`.
- There are no class methods to be mapped to `run`.