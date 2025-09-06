Based on the provided API documentation, here is the classification of the functions and methods:

### Top-Level Functions
These are the functions that are not part of any class:
1. `load_module_gpu`
2. `unload_module_gpu`
3. `initial_model_load`
4. `preprocess_video`
5. `do_sample`
6. `run_img2vid`
7. `load_model`

### Methods
There are no methods defined in the provided documentation, as all functions are defined as top-level functions without being part of any class. Therefore, there are no associated classes or method types (static, instance, or class methods).

### Total Number of Interface Classes
There are **0 interface classes** mentioned in the provided documentation.

In summary:
- **Top-Level Functions**: 7
- **Methods**: 0
- **Total Number of Interface Classes**: 0

Here’s how to fill in the template based on the provided API documentation and the questions you asked:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
Based on the provided documentation, there are no interface classes mentioned explicitly. The functions listed are all top-level functions, meaning that there are no class instances that need to be initialized in the `create_interface_objects` method. Therefore, **initialization is unnecessary** for any interface class objects.

### Q2: Which top-level functions should be mapped to `run`?
The following top-level functions from the interface documentation should be mapped to the `run` method:
1. `load_module_gpu`
2. `unload_module_gpu`
3. `initial_model_load`
4. `preprocess_video`
5. `do_sample`
6. `run_img2vid`
7. `load_model`

Each of these functions can be invoked using the format `run(function_name, **kwargs)`.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`?
Since the provided documentation does not specify any classes with instance methods, class methods, or static methods, we can assume there are no such methods to map. All methods are top-level functions, and there are no specific classes mentioned that contain methods to be called.

However, if there were classes defined in the documentation, you would map their methods to the `run` method using the format `run(class_name_method_name, **kwargs)`. Since there are no classes with methods mentioned, there is nothing to map in this context.

### Summary of Mapping
- **Initialization in `create_interface_objects`**: Unnecessary (no interface class objects).
- **Top-Level Functions to Map in `run`**: All top-level functions (7 total).
- **Instance/Static/Class Methods to Map in `run`**: None (no methods defined in classes). 

This should guide you on how to fill in the `CustomAdapter` class based on the provided template.