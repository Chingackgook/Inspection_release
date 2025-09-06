# API Documentation for EnSpellCorrector

## Class: EnSpellCorrector

### Description
The `EnSpellCorrector` class provides functionality for correcting English spelling errors based on a frequency dictionary of words. It uses a probabilistic approach to suggest corrections for misspelled words.

### Attributes
- `word_freq_dict`: A dictionary containing word frequencies, where the key is the word and the value is its frequency (int).
- `custom_confusion_dict`: A dictionary for custom spelling corrections, where the key is the misspelled word and the value is the correct word.
- `sum_freq`: The total frequency of all words in the `word_freq_dict`.

### Method: __init__
```python
def __init__(self, word_freq_dict: dict = None, custom_confusion_dict: dict = None, en_dict_path: str = None)
```
#### Parameters
- `word_freq_dict` (dict, optional): A dictionary of word frequencies. If provided, `en_dict_path` cannot be set. Default is `None`.
- `custom_confusion_dict` (dict, optional): A dictionary for custom spelling corrections. Default is `None`.
- `en_dict_path` (str, optional): Path to a gzip-compressed JSON file containing a word frequency dictionary. Default is `None`.

#### Return Value
None

#### Description
Initializes the `EnSpellCorrector` instance. Loads the word frequency dictionary from the specified path or uses the default if none is provided.

---

### Method: edits1
```python
def edits1(word: str) -> set
```
#### Parameters
- `word` (str): The word for which to generate edits.

#### Return Value
- `set`: A set of all edits that are one edit away from the input word.

#### Description
Generates all possible edits that are one character modification away from the given word (deletes, transposes, replaces, and inserts).

---

### Method: edits2
```python
def edits2(word: str) -> set
```
#### Parameters
- `word` (str): The word for which to generate edits.

#### Return Value
- `generator`: A generator yielding all edits that are two edits away from the input word.

#### Description
Generates all possible edits that are two character modifications away from the given word by applying the `edits1` function twice.

---

### Method: known
```python
def known(word_freq_dict: dict) -> set
```
#### Parameters
- `word_freq_dict` (dict): A dictionary of words to check against the known words in the frequency dictionary.

#### Return Value
- `set`: A set of words from `word_freq_dict` that are known (i.e., present in the frequency dictionary).

#### Description
Identifies the subset of words from the provided dictionary that are present in the known word frequency dictionary.

---

### Method: probability
```python
def probability(word: str) -> float
```
#### Parameters
- `word` (str): The word for which to calculate the probability.

#### Return Value
- `float`: The probability of the word based on its frequency in the dictionary.

#### Description
Calculates the probability of a word based on its frequency relative to the total frequency of all words in the dictionary.

---

### Method: candidates
```python
def candidates(word: str) -> set
```
#### Parameters
- `word` (str): The word for which to generate possible corrections.

#### Return Value
- `set`: A set of possible spelling corrections for the input word.

#### Description
Generates a set of possible spelling corrections for the given word by checking known words, one-edit-away edits, and two-edit-away edits.

---

### Method: correct_word
```python
def correct_word(word: str) -> str
```
#### Parameters
- `word` (str): The word to be corrected.

#### Return Value
- `str`: The most probable spelling correction for the input word.

#### Description
Determines the most likely spelling correction for the given word based on the candidates generated and their probabilities.

---

### Method: set_en_custom_confusion_dict
```python
def set_en_custom_confusion_dict(path: str)
```
#### Parameters
- `path` (str): The path to a file containing custom confusion pairs.

#### Return Value
None

#### Description
Loads a custom confusion dictionary from the specified file, allowing for additional spelling corrections beyond the default dictionary.

---

### Method: correct
```python
def correct(sentence: str, include_symbol: bool = True) -> dict
```
#### Parameters
- `sentence` (str): The input sentence to be corrected.
- `include_symbol` (bool, optional): Whether to include symbols in the correction process. Default is `True`.

#### Return Value
- `dict`: A dictionary containing the original sentence, the corrected sentence, and a list of errors found.

#### Description
Corrects spelling errors in the provided sentence and returns a structured result including the source sentence, corrected sentence, and details of the corrections made.

---

### Method: correct_batch
```python
def correct_batch(sentences: List[str], **kwargs) -> List[dict]
```
#### Parameters
- `sentences` (List[str]): A list of sentences to be corrected.
- `**kwargs`: Additional parameters to be passed to the `correct` method.

#### Return Value
- `List[dict]`: A list of dictionaries, each containing the original sentence, the corrected sentence, and details of the errors found.

#### Description
Processes a batch of sentences for spelling correction, returning a list of results for each sentence.

