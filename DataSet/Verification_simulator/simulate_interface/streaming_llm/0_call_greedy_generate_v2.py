from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.streaming_llm import *
exe = Executor('streaming_llm', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/streaming-llm/examples/run_streaming_llama.py'
import warnings
import torch
import json
import os
import time
import re
import sys
from tqdm import tqdm
from streaming_llm.utils import load
from streaming_llm.utils import download_url
from streaming_llm.utils import load_jsonl
from streaming_llm.enable_streaming_llm import enable_streaming_llm
warnings.filterwarnings('ignore')

@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = None
    for (idx, prompt) in enumerate(prompts):
        prompt = 'USER: ' + prompt + '\n\nASSISTANT: '
        print('\n' + prompt, end='')
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        input_ids = input_ids.to(model.device)
        seq_len = input_ids.shape[1]
        if kv_cache is not None:
            space_needed = seq_len + max_gen_len
            past_key_values = kv_cache.evict_for_space(past_key_values, space_needed)
        past_key_values = exe.run('greedy_generate', model=model, tokenizer=tokenizer, input_ids=input_ids, past_key_values=past_key_values, max_gen_len=max_gen_len)

def run_inference():
    model_name_or_path = 'lmsys/vicuna-7b-v1.5'
    data_root = 'data/'
    test_filepath = os.path.join(data_root, 'mt_bench.jsonl')
    print(f'Loading data from {test_filepath} ...')
    if not os.path.exists(test_filepath):
        download_url('https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl', data_root)
        os.rename(os.path.join(data_root, 'question.jsonl'), test_filepath)
    (model, tokenizer) = load(model_name_or_path)
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample['turns']
    kv_cache = enable_streaming_llm(model, start_size=4, recent_size=2000)
    streaming_inference(model, tokenizer, prompts, kv_cache)
run_inference()