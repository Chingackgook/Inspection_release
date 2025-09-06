from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.fish_speech_fixed import *
exe = Executor('fish_speech_fixed','simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package
import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union
import click
import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from fish_speech.conversation import CODEBOOK_PAD_TOKEN_ID
from fish_speech.conversation import Conversation
from fish_speech.conversation import Message
from fish_speech.conversation import TextPart
from fish_speech.conversation import VQPart
from fish_speech.models.text2semantic.llama import BaseModelArgs
from fish_speech.text import clean_text
from fish_speech.text import split_text
from fish_speech.tokenizer import IM_END_TOKEN
from fish_speech.tokenizer import FishTokenizer
from torch.nn.attention import SDPBackend
from torch.nn.attention import sdpa_kernel
from fish_speech.models.text2semantic.llama import BaseTransformer
from fish_speech.models.text2semantic.llama import DualARTransformer
from fish_speech.models.text2semantic.llama import NaiveTransformer
import traceback
# end

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
if hasattr(torch._inductor.config, 'fx_graph_cache'):
    torch._inductor.config.fx_graph_cache = True
from torch.nn.attention import SDPBackend, sdpa_kernel
from fish_speech.models.text2semantic.llama import BaseTransformer, DualARTransformer, NaiveTransformer

text = '你说的对, 但是原神是一款由米哈游自主研发的开放世界手游.'
prompt_text = None
prompt_tokens = None

# Parts that may need manual modification:
num_samples = 1
max_new_tokens = 0
top_p = 0.7
repetition_penalty = 1.5
temperature = 0.7
checkpoint_path = 'checkpoints/fish-speech-1.5'
device = 'cuda'
compile = False
seed = 42
half = False
iterative_prompt = True
chunk_length = 100
output_dir = 'temp'  # This will be replaced with FILE_RECORD_PATH
# end

os.makedirs(output_dir, exist_ok=True)
precision = torch.half if half else torch.bfloat16

if prompt_text is not None and len(prompt_text) != len(prompt_tokens):
    raise ValueError(f'Number of prompt text ({len(prompt_text)}) and prompt tokens ({len(prompt_tokens)}) should be the same')

logger.info('Loading model ...')
t0 = time.time()
model, decode_one_token = exe.run('load_model', checkpoint_path=checkpoint_path, device=device, precision=precision, compile=compile)

with torch.device(device):
    model.setup_caches(max_batch_size=1, max_seq_len=model.config.max_seq_len, dtype=next(model.parameters()).dtype)

if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info(f'Time to load model: {time.time() - t0:.02f} seconds')

if prompt_tokens is not None:
    prompt_tokens = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

generator = exe.run('generate_long', model=model, device=device, decode_one_token=decode_one_token, text=text, num_samples=num_samples, max_new_tokens=max_new_tokens, top_p=top_p, repetition_penalty=repetition_penalty, temperature=temperature, compile=compile, iterative_prompt=iterative_prompt, chunk_length=chunk_length, prompt_text=prompt_text, prompt_tokens=prompt_tokens)

idx = 0
codes = []
for response in generator:
    if response.action == 'sample':
        codes.append(response.codes)
        logger.info(f'Sampled text: {response.text}')
    elif response.action == 'next':
        if codes:
            # Replace output path with FILE_RECORD_PATH
            codes_npy_path = os.path.join(FILE_RECORD_PATH, f'codes_{idx}.npy')
            np.save(codes_npy_path, torch.cat(codes, dim=1).cpu().numpy())
            logger.info(f'Saved codes to {codes_npy_path}')
        logger.info(f'Next sample')
        codes = []
        idx += 1
    else:
        logger.error(f'Error: {response}')

# Directly run the main logic
