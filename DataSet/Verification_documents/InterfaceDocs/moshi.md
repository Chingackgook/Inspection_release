```markdown
# API Documentation

## Function: `scatter_with_mask_`

### Description
Scatters values into a tensor at specified indices, while skipping updates that are masked.

### Parameters
- **tensor** (`torch.Tensor`): The tensor to scatter values into.
- **dim** (`int`): The dimension along which to scatter.
- **index** (`torch.Tensor`): The indices at which to scatter the values.
- **value** (`torch.Tensor`): The values to scatter.
- **mask** (`torch.Tensor`): A boolean tensor indicating which updates to skip.

### Return Value
- **None**: This function modifies the input tensor in place.

### Purpose
This function allows for selective updates to a tensor based on a mask, enabling more controlled data manipulation during model operations.

---

## Class: `LMGen`

### Description
`LMGen` is a streaming module for generating language model outputs, supporting various configurations for sampling and conditioning.

### Attributes
- **lm_model** (`LMModel`): The language model used for generation.
- **use_sampling** (`bool`): Flag indicating whether to use sampling for token generation.
- **temp** (`float`): Temperature parameter for sampling.
- **temp_text** (`float`): Temperature parameter for text sampling.
- **top_k** (`int`): The number of top tokens to consider during sampling.
- **top_k_text** (`int`): The number of top text tokens to consider during sampling.
- **cfg_coef** (`float`): Coefficient for conditioning.
- **check** (`bool`): Flag for enabling checks during generation.
- **condition_tensors** (`ConditionTensors | None`): Tensors used for conditioning, if applicable.
- **on_text_hook** (`Optional[Callable[[torch.Tensor], None]]`): Hook function to call with generated text tokens.
- **on_text_logits_hook** (`Optional[Callable[[torch.Tensor], None]]`): Hook function to call with text logits.
- **on_audio_hook** (`Optional[Callable[[torch.Tensor], None]]`): Hook function to call with audio tokens.
- **support_out_of_sync** (`bool`): Flag indicating support for out-of-sync execution.
- **cfg_is_masked_until** (`list[int] | None`): List indicating masking conditions for CFG.
- **cfg_is_no_text** (`bool`): Flag indicating whether to disable text conditioning.

### Method: `__init__`

#### Description
Initializes the `LMGen` instance with the specified parameters.

#### Parameters
- **lm_model** (`LMModel`): The language model to use for generation.
- **use_sampling** (`bool`, optional): Default is `True`. Indicates whether to use sampling.
- **temp** (`float`, optional): Default is `0.8`. Temperature for sampling.
- **temp_text** (`float`, optional): Default is `0.7`. Temperature for text sampling.
- **top_k** (`int`, optional): Default is `250`. Top K tokens for sampling.
- **top_k_text** (`int`, optional): Default is `25`. Top K text tokens for sampling.
- **cfg_coef** (`float`, optional): Default is `1.0`. Coefficient for conditioning.
- **check** (`bool`, optional): Default is `False`. Enables checks during generation.
- **condition_tensors** (`ConditionTensors | None`, optional): Default is `None`. Tensors for conditioning.
- **on_text_hook** (`Optional[Callable[[torch.Tensor], None]]`, optional): Default is `None`. Hook for text tokens.
- **on_text_logits_hook** (`Optional[Callable[[torch.Tensor], None]]`, optional): Default is `None`. Hook for text logits.
- **on_audio_hook** (`Optional[Callable[[torch.Tensor], None]]`, optional): Default is `None`. Hook for audio tokens.
- **support_out_of_sync** (`bool`, optional): Default is `False`. Indicates support for out-of-sync execution.
- **cfg_is_masked_until** (`list[int] | None`, optional): Default is `None`. Masking conditions for CFG.
- **cfg_is_no_text** (`bool`, optional): Default is `False`. Disables text conditioning.

#### Return Value
- **None**: Initializes the instance.

#### Purpose
To create an instance of `LMGen` with the specified configuration for language model generation.

### Method: `step`

#### Description
Performs a single step of generation using the provided input tokens.

#### Parameters
- **input_tokens** (`torch.Tensor`): Input tokens for the generation step, shape `[B, K, T]`.
- **depformer_replace_tokens** (`torch.Tensor | None`, optional): Tokens to replace in the depformer, default is `None`.

#### Return Value
- **torch.Tensor | None**: The generated output tokens, or `None` if not applicable.

#### Purpose
To generate the next set of tokens based on the input tokens and the current state of the model.

### Method: `step_with_extra_heads`

#### Description
Performs a generation step and returns additional outputs from extra heads of the model.

#### Parameters
- **input_tokens** (`torch.Tensor`): Input tokens for the generation step, shape `[B, K, T]`.
- **depformer_replace_tokens** (`torch.Tensor | None`, optional): Tokens to replace in the depformer, default is `None`.

#### Return Value
- **tuple[torch.Tensor, list[torch.Tensor]] | None**: The generated output tokens and a list of outputs from extra heads, or `None` if not applicable.

#### Purpose
To generate tokens while also retrieving additional information from the model's extra heads.

### Method: `depformer_step`

#### Description
Executes a step in the depformer, generating tokens based on the previous token and transformer output.

#### Parameters
- **text_token** (`torch.Tensor`): The current text token, shape `[B]`.
- **transformer_out** (`torch.Tensor`): The output from the transformer, shape `[B, ...]`.

#### Return Value
- **torch.Tensor**: The generated depformer tokens, shape `[B, lm_model.dep_q]`.

#### Purpose
To generate tokens in the depformer based on the current text token and transformer output.
```

