# API Documentation

## Class: VALLE

### Description
The `VALLE` class implements the VALLE model as described in the paper "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers" (https://arxiv.org/abs/2301.02111). It is designed for text-to-speech synthesis using neural codec language models.

### Attributes
- `language_ID`: A dictionary mapping language codes (e.g., 'en', 'zh', 'ja') to their corresponding integer IDs.
- `ar_language_embedding`: A token embedding layer for autoregressive language embeddings.
- `nar_language_embedding`: A token embedding layer for non-autoregressive language embeddings.

### Method: __init__

#### Description
Initializes the VALLE model with specified parameters.

#### Parameters
- `d_model` (int): The number of expected features in the input (required).
- `nhead` (int): The number of heads in the multihead attention models (required).
- `num_layers` (int): The number of sub-decoder layers in the decoder (required).
- `norm_first` (bool, optional): Whether to apply layer normalization before or after the attention and feed-forward layers. Default is `True`.
- `add_prenet` (bool, optional): Whether to add a prenet layer. Default is `False`.
- `prefix_mode` (int, optional): Mode for prefix handling. Default is `0`.
- `share_embedding` (bool, optional): Whether to share embeddings between different components. Default is `True`.
- `nar_scale_factor` (float, optional): Scale factor for non-autoregressive components. Default is `1.0`.
- `**kwargs`: Additional keyword arguments for further customization.

### Method: forward

#### Description
Defines the forward pass of the model. This method is not implemented and raises a `NotImplementedError`.

#### Parameters
- `x` (torch.Tensor): A 2-D tensor of shape (1, S) representing the input features.
- `x_lens` (torch.Tensor): A 1-D tensor of shape (1,) containing the number of tokens in `x` before padding.
- `y` (Union[torch.Tensor, PromptedFeatures]): A 3-D tensor of shape (1, T, 8) representing the target features.
- `y_lens` (Union[torch.Tensor, PromptedFeatures]): A 1-D tensor or PromptedFeatures object containing the lengths of the target features.
- `reduction` (str, optional): Specifies the reduction method to apply to the output. Default is `"sum"`.
- `train_stage` (int, optional): Indicates the training stage. Default is `0`.
- `**kwargs`: Additional keyword arguments.

#### Returns
- Raises `NotImplementedError`.

### Method: inference

#### Description
Generates audio code matrix predictions based on the input text and enrolled features.

#### Parameters
- `x` (torch.Tensor): A 2-D tensor of shape (1, S) representing the input features.
- `x_lens` (torch.Tensor): A 1-D tensor of shape (1,) containing the number of tokens in `x` before padding.
- `y` (torch.Tensor): A 3-D tensor of shape (1, T, 8) representing the target features.
- `enroll_x_lens` (torch.Tensor): A 1-D tensor containing the lengths of the enrolled features.
- `top_k` (int, optional): The number of highest probability tokens to keep for top-k filtering. Default is `-100`.
- `temperature` (float, optional): The value used to modulate the next token probabilities. Must be strictly positive. Default is `1.0`.
- `prompt_language` (str, optional): The language of the prompt text. Default is `None`.
- `text_language` (Union[str, List], optional): The language of the text input. Can be a string or a list of strings. Default is `None`.
- `best_of` (int, optional): The number of best sequences to return. Default is `1`.
- `length_penalty` (float, optional): Penalty for longer sequences. Default is `1.0`.
- `return_worst` (bool, optional): Whether to return the worst sequence instead of the best. Default is `False`.

#### Returns
- `torch.Tensor`: The predicted audio code matrix.

### Method: continual

#### Description
Generates audio code matrix predictions in a continual manner based on the input text and enrolled features.

#### Parameters
- `x` (torch.Tensor): A 2-D tensor of shape (1, S) representing the input features.
- `x_lens` (torch.Tensor): A 1-D tensor of shape (1,) containing the number of tokens in `x` before padding.
- `y` (torch.Tensor): A 3-D tensor of shape (1, T, 8) representing the target features.

#### Returns
- `torch.Tensor`: The predicted audio code matrix.

## Function: top_k_top_p_filtering

#### Description
Filters a distribution of logits using top-k and/or nucleus (top-p) filtering.

#### Parameters
- `logits` (torch.Tensor): Logits distribution shape (batch size, vocabulary size).
- `top_k` (int, optional): Keep only the top k tokens with the highest probability (top-k filtering). Default is `0`.
- `top_p` (float, optional): Keep the top tokens with cumulative probability >= top_p (nucleus filtering). Default is `1.0`.
- `filter_value` (float, optional): Value to assign to filtered logits. Default is `-float("Inf")`.
- `min_tokens_to_keep` (int, optional): Minimum number of tokens to keep in the output. Default is `1`.

#### Returns
- `torch.Tensor`: The filtered logits.

#### Note
- If `top_k > 0`, only the top k tokens with the highest probability are kept.
- If `top_p < 1.0`, tokens with cumulative probability above the threshold are removed, ensuring at least `min_tokens_to_keep` tokens are retained.

