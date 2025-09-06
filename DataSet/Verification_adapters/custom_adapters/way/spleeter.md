Based on the provided API documentation, here is a clear classification of the components:

### Top-Level Functions
- **`create_estimator`**: This is a top-level function that initializes a TensorFlow estimator for source separation.

### Methods and Their Class Associations
1. **`__init__`**: 
   - **Class**: `Separator`
   - **Type**: Instance method

2. **`join`**: 
   - **Class**: `Separator`
   - **Type**: Instance method

3. **`separate`**: 
   - **Class**: `Separator`
   - **Type**: Instance method

4. **`separate_to_file`**: 
   - **Class**: `Separator`
   - **Type**: Instance method

5. **`save_to_file`**: 
   - **Class**: `Separator`
   - **Type**: Instance method

### Total Number of Interface Classes
- There is **1 interface class**: `Separator`. 

In summary:
- **Top-Level Functions**: 1 (`create_estimator`)
- **Methods**: 5 (all belonging to the `Separator` class)
- **Total Number of Interface Classes**: 1 (`Separator`)

Let's address your questions one by one:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary? (For top-level functions, initialization is not needed)

In the provided interface documentation, there is only one interface class: `Separator`. Therefore, you need to initialize an object of the `Separator` class in the `create_interface_objects` method. The initialization is necessary because the methods of the `Separator` class will be called later in the `run` method.

### Q2: Which top-level functions should be mapped to `run`?

The top-level function mentioned in the interface documentation is `create_estimator`. However, since this is a top-level function, it does not require initialization of an object and should not be included in the `run` method. Therefore, **no top-level functions** should be mapped to `run`.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`? (Do not omit methods that the documentation mentioned)

Based on the interface documentation for the `Separator` class, the following instance methods should be mapped to `run`:

1. **`separate`**: This method can be called using the dispatch key `separate`.
2. **`separate_to_file`**: This method can be called using the dispatch key `separate_to_file`.
3. **`save_to_file`**: This method can be called using the dispatch key `save_to_file`.
4. **`join`**: This method can be called using the dispatch key `join`.

Since there is only one interface class (`Separator`), you can directly map the methods as follows:
- `run('separate', **kwargs)`
- `run('separate_to_file', **kwargs)`
- `run('save_to_file', **kwargs)`
- `run('join', **kwargs)`

In summary:
- **Initialization in `create_interface_objects`**: `Separator` class object.
- **Top-level functions mapped to `run`**: None.
- **Instance methods mapped to `run`**: `separate`, `separate_to_file`, `save_to_file`, `join`.