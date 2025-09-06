Based on the provided API documentation, we can classify the components as follows:

### Top-Level Functions
- `greedy_generate`

### Methods
There are no specific methods mentioned in the provided documentation, as it only details the `greedy_generate` function without any associated classes or their methods.

### Total Number of Interface Classes
- There are **0** interface classes mentioned in the provided documentation.

### Summary
- **Top-Level Functions**: 1 (`greedy_generate`)
- **Methods**: None
- **Total Number of Interface Classes**: 0

Sure! Let's address your questions one by one:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary? (For top-level functions, initialization is not needed)

In the provided API documentation, there are no specific interface classes mentioned. The documentation only describes a top-level function `greedy_generate`. Therefore, there is no need to initialize any interface class objects in `create_interface_objects`, as it is unnecessary. You can simply pass when the `interface_class_name` is not required.

### Q2: Which top-level functions should be mapped to `run`?

The only top-level function mentioned in the documentation is `greedy_generate`. Therefore, you should map this function to the `run` method in the following way:

- `run('greedy_generate', **kwargs)`

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`? (Do not omit methods that the documentation mentioned)

Based on the provided API documentation, there are no specific instance methods, class methods, or static methods mentioned for any classes. The only function documented is `greedy_generate`, which is a top-level function.

If you were to create a class that includes methods related to `greedy_generate`, you would need to map them similarly. However, since the documentation does not provide any such methods, you do not have any instance methods or class/static methods to map in this case.

### Summary of Answers:
- **Q1**: No interface class objects need to be initialized in `create_interface_objects`.
- **Q2**: Map `greedy_generate` to `run`: `run('greedy_generate', **kwargs)`.
- **Q3**: No instance, class, or static methods to map as none are provided in the documentation.