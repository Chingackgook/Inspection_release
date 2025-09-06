Based on the provided API documentation for the `vis` function, here’s the classification of the elements:

### Top-Level Functions
- `vis`: This is a top-level function that takes parameters related to image visualization.

### Methods
There are no methods explicitly defined in the provided documentation, as it only describes a single function (`vis`). Therefore, there are no instance methods, class methods, or static methods to classify.

### Total Number of Interface Classes
There are no interface classes mentioned in the provided documentation. The documentation only describes the `vis` function, and there is no indication of any classes being defined.

### Summary
- **Top-Level Functions**: 1 (vis)
- **Methods**: 0
- **Total Number of Interface Classes**: 0

If there are additional classes or methods not included in the provided documentation, please provide that information for a more comprehensive classification.

Certainly! Here’s how to fill in the template based on your questions:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
In the provided documentation, there are no interface classes mentioned that require initialization. Since the `vis` function is a top-level function and does not belong to any class, you do not need to initialize any interface class objects in the `create_interface_objects` method. Therefore, initialization is unnecessary.

### Q2: Which top-level functions should be mapped to `run`?
The only top-level function mentioned in the documentation is the `vis` function. You should map this function in the `run` method. The mapping will look like this:
- `run('vis', **kwargs)`: This will call the `vis` function with the provided keyword arguments.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`? (Do not omit methods that the documentation mentioned)
Based on the provided documentation, there are no instance methods, class methods, or static methods explicitly mentioned. The only function described is the `vis` function, which is a top-level function. Therefore, you will not need to map any additional methods to the `run` method aside from the `vis` function.

### Summary of Mappings for `run`:
- **Top-Level Function**: 
  - `vis` should be directly mapped as `run('vis', **kwargs)`.

- **Interface Class Methods**: 
  - None to map, as there are no interface classes provided.

- **Instance/Static/Class Methods**: 
  - None to map, as there are no methods from classes provided in the documentation.

In conclusion, in the `run` method, you will primarily handle the `vis` function and its corresponding parameters without needing to initialize any class objects or map other methods.