# API Documentation

## Function: `keywords`

### Description
Extracts keywords from a given sentence using the Jieba Tokenizer.

### Parameters
- **sentence** (str): The input sentence from which keywords are to be extracted.
- **topK** (int, optional): The number of top keywords to return. Default is 5.
- **withWeight** (bool, optional): If True, returns the weight of each keyword. Default is False.
- **allowPOS** (tuple, optional): A tuple of part-of-speech tags to filter the keywords. Default is an empty tuple.

### Returns
- **list**: A list of keywords extracted from the input sentence.

---

## Function: `sv`

### Description
Obtains a vector representation of a segmented sentence, where the vector is composed of the individual word vectors.

### Parameters
- **sentence** (str): The input sentence, which should be segmented and joined by spaces.
- **ignore** (bool, optional): If True, ignores out-of-vocabulary (OOV) words. If False, generates a random vector for OOV words. Default is False.

### Returns
- **list**: A list of vectors corresponding to each word in the input sentence.

---

## Function: `bow`

### Description
Obtains a vector representation of a segmented sentence using the Bag of Words (BoW) approach.

### Parameters
- **sentence** (str): The input sentence, which should be segmented and joined by spaces.
- **ignore** (bool, optional): If True, ignores out-of-vocabulary (OOV) words. If False, generates a random vector for OOV words. Default is False.

### Returns
- **numpy.ndarray**: A single vector representing the input sentence in the BoW format.

---

## Function: `v`

### Description
Obtains the vector representation of a specific word. Raises an exception if the word is out-of-vocabulary (OOV).

### Parameters
- **word** (str): The input word for which the vector is to be obtained.

### Returns
- **numpy.ndarray**: The vector representation of the input word.

### Raises
- **KeyError**: If the word is out-of-vocabulary (OOV).

---

## Function: `nearby`

### Description
Finds nearby words (synonyms) for a given word based on the word embedding model.

### Parameters
- **word** (str): The input word for which nearby words are to be found.
- **size** (int, optional): The number of nearby words to return. Default is 10.

### Returns
- **tuple**: A tuple containing two lists:
  - **list**: A list of nearby words.
  - **list**: A list of scores corresponding to the nearby words.

---

## Function: `compare`

### Description
Compares the similarity between two sentences.

### Parameters
- **s1** (str): The first sentence to compare.
- **s2** (str): The second sentence to compare.
- **seg** (bool, optional): If True, the original sentences will be segmented. If False, the sentences are assumed to be pre-segmented. Default is True.
- **ignore** (bool, optional): If True, ignores out-of-vocabulary (OOV) words. If False, generates a random vector for OOV words. Default is False.
- **stopwords** (bool, optional): If True, stopwords will be ignored in the comparison. Default is False.

### Returns
- **float**: A similarity score between 0.0 and 1.0, where 1.0 indicates identical sentences.

---

## Function: `describe`

### Description
Provides summary information about the vector model, including vocabulary size and model path.

### Returns
- **dict**: A dictionary containing:
  - **vocab_size** (int): The size of the vocabulary in the vector model.
  - **version** (str): The version of the model.
  - **model_path** (str): The path to the model file.

---

## Function: `display`

### Description
Displays the nearby words (synonyms) for a given word.

### Parameters
- **word** (str): The input word for which nearby words are to be displayed.
- **size** (int, optional): The number of nearby words to display. Default is 10.

### Returns
- **None**: This function prints the nearby words to the console.

---

## Function: `main`

### Description
The main entry point of the script. Displays nearby words for specific examples.

### Parameters
- **None**

### Returns
- **None**: This function does not return any value; it prints output to the console.

--- 

This documentation provides a comprehensive overview of the functions available in the API, including their parameters, return values, and purposes.

