Based on the provided API documentation, here is the classification of the functions and methods:

### Top-Level Functions
These are standalone functions that are not part of any class:
1. `process_folder`
2. `extract_video`
3. `cut_video`
4. `denoise_image_sequence`
5. `video_from_sequence`

### Methods
There are no methods defined in the provided documentation, as it does not mention any classes or their associated methods. 

### Total Number of Interface Classes
There are **0 interface classes** mentioned in the provided documentation. All functions are presented as top-level functions without any indication of being part of a class structure.

### Summary
- **Top-Level Functions**: 5
- **Methods**: 0
- **Total Number of Interface Classes**: 0

Sure! Let's address your questions one by one.

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
Since the provided documentation does not mention any specific classes or require any initialization for interface class objects, you can conclude that initialization is unnecessary for the functions mentioned. The functions are all top-level functions, and therefore, there are no class instances that need to be created within `create_interface_objects`.

### Q2: Which top-level functions should be mapped to `run`?
The top-level functions from the interface documentation that should be mapped to `run` are:
1. `process_folder`
2. `extract_video`
3. `cut_video`
4. `denoise_image_sequence`
5. `video_from_sequence`

These functions can be called directly in the `run` method using the `dispatch_key` parameter to determine which function to execute.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`?
According to the documentation provided, there are no specific classes with instance methods, class methods, or static methods mentioned. Therefore, there are no methods that need to be mapped in the `run` method. 

In summary:
- **Top-Level Functions to Map**: `process_folder`, `extract_video`, `cut_video`, `denoise_image_sequence`, `video_from_sequence`
- **No Instance/Static/Class Methods to Map**: There are no classes or their associated methods to map in this context.

This means that your `run` method will primarily handle the execution of the top-level functions based on the `dispatch_key` provided, and no additional class-based methods need to be considered.