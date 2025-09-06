Based on the provided API documentation, here is the classification of the interface components:

### Top-Level Functions
There are no explicit top-level functions mentioned in the documentation. All functions are methods belonging to the `EnSpellCorrector` class.

### Methods
All methods listed belong to the `EnSpellCorrector` class and are instance methods. Here is the breakdown:

1. **Instance Methods**:
   - `__init__(self, word_freq_dict: dict = None, custom_confusion_dict: dict = None, en_dict_path: str = None)`
   - `edits1(self, word: str) -> set`
   - `edits2(self, word: str) -> set`
   - `known(self, word_freq_dict: dict) -> set`
   - `probability(self, word: str) -> float`
   - `candidates(self, word: str) -> set`
   - `correct_word(self, word: str) -> str`
   - `set_en_custom_confusion_dict(self, path: str)`
   - `correct(self, sentence: str, include_symbol: bool = True) -> dict`
   - `correct_batch(self, sentences: List[str], **kwargs) -> List[dict]`

### Total Number of Interface Classes
There is **1 interface class** identified in the documentation, which is `EnSpellCorrector`.

Let's address your questions one by one:

### Q1: Which interface class objects need to be initialized in `create_interface_objects`, or is initialization unnecessary?
In the `create_interface_objects` method, you need to initialize an instance of the `EnSpellCorrector` class, as it is the only interface class mentioned in the documentation. The initialization is necessary to create an object that can be used to call its methods later in the `run` method. If you have a custom confusion dictionary to load, you can also pass it as a parameter during initialization.

### Q2: Which top-level functions should be mapped to `run`?
There are no top-level functions mentioned in the interface documentation that need to be mapped to `run`. All functions are methods of the `EnSpellCorrector` class.

### Q3: Which instance methods, class methods, or static methods should be mapped to `run`?
The following methods from the `EnSpellCorrector` class should be mapped to the `run` method:

1. **Instance Methods**:
   - `correct(sentence: str, include_symbol: bool = True)`: This should be mapped as `run('correct', **kwargs)`.
   - `correct_batch(sentences: List[str], **kwargs)`: This should be mapped as `run('correct_batch', **kwargs)`.
   - `set_en_custom_confusion_dict(path: str)`: This should be mapped as `run('set_en_custom_confusion_dict', **kwargs)`.
   - `correct_word(word: str)`: This should be mapped as `run('correct_word', **kwargs)`.
   - `candidates(word: str)`: This should be mapped as `run('candidates', **kwargs)`.
   - `probability(word: str)`: This should be mapped as `run('probability', **kwargs)`.
   - `known(word_freq_dict: dict)`: This should be mapped as `run('known', **kwargs)`.
   - `edits1(word: str)`: This should be mapped as `run('edits1', **kwargs)`.
   - `edits2(word: str)`: This should be mapped as `run('edits2', **kwargs)`.

In summary, the `run` method will handle the execution of these instance methods based on the `dispatch_key` provided, which corresponds to the method names in the `EnSpellCorrector` class. If there is only one interface class, you can directly map the method names without prefixing them with the class name.