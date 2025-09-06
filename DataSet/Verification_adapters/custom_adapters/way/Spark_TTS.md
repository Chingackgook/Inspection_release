Based on the provided API documentation for the `SparkTTS` class, here is the classification of the functions and methods:

### Top-Level Functions:
- There are no top-level functions mentioned in the provided documentation. All functions are methods belonging to the `SparkTTS` class.

### Methods and Their Classification:
1. **Method:** `__init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0"))`
   - **Class:** `SparkTTS`
   - **Type:** Instance Method

2. **Method:** `process_prompt(self, text: str, prompt_speech_path: Path, prompt_text: str = None) -> Tuple[str, torch.Tensor]`
   - **Class:** `SparkTTS`
   - **Type:** Instance Method

3. **Method:** `process_prompt_control(self, gender: str, pitch: str, speed: str, text: str) -> str`
   - **Class:** `SparkTTS`
   - **Type:** Instance Method

4. **Method:** `inference(self, text: str, prompt_speech_path: Path = None, prompt_text: str = None, gender: str = None, pitch: str = None, speed: str = None, temperature: float = 0.8, top_k: float = 50, top_p: float = 0.95) -> torch.Tensor`
   - **Class:** `SparkTTS`
   - **Type:** Instance Method

### Total Number of Interface Classes:
- There is **1 interface class** identified in the documentation, which is `SparkTTS`. 

### Summary:
- **Top-Level Functions:** 0
- **Methods:** 4 (all instance methods of `SparkTTS`)
- **Total Number of Interface Classes:** 1 (`SparkTTS`)

Let's address your questions one by one:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary? (For top-level functions, initialization is not needed.)

In the context of the `SparkTTS` class provided in the interface documentation, the following class object needs to be initialized in `create_interface_objects`:

- **`SparkTTS`**: This is the main interface class that provides the text-to-speech functionality. You will need to create an instance of `SparkTTS` using the appropriate parameters (like `model_dir` and `device`) when the `interface_class_name` matches `SparkTTS`.

Initialization is necessary because you will need an instance of `SparkTTS` to call its methods in the `run` method.

### Q2: Which top-level functions should be mapped to `run`?

There are no top-level functions mentioned in the provided interface documentation, so there are no top-level functions to map to `run`. All relevant functionality is encapsulated within the methods of the `SparkTTS` class.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`? (Do not omit methods that the documentation mentioned.)

Based on the methods described in the `SparkTTS` class, the following methods should be mapped to `run`:

1. **Instance Methods**:
   - **`inference`**: This method can be called as `run('inference', **kwargs)`, where `kwargs` includes parameters like `text`, `prompt_speech_path`, `prompt_text`, `gender`, `pitch`, `speed`, `temperature`, `top_k`, and `top_p`.

2. **Other Instance Methods**:
   - **`process_prompt`**: This method can be called as `run('process_prompt', **kwargs)`, where `kwargs` includes parameters like `text`, `prompt_speech_path`, and `prompt_text`.
   - **`process_prompt_control`**: This method can be called as `run('process_prompt_control', **kwargs)`, where `kwargs` includes parameters like `gender`, `pitch`, `speed`, and `text`.

Since there is only one interface class (`SparkTTS`), you can directly map the instance methods without needing to specify the class name. 

In summary:
- For the `run` method:
  - Map `inference`, `process_prompt`, and `process_prompt_control` as `run(method_name, **kwargs)` directly.