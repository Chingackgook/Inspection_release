from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.gpt_fast import *
exe = Executor('gpt_fast', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/gpt-fast/generate.py'
import itertools
import sys
import time
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import create_block_mask
from model import Transformer
from tokenizer import get_tokenizer
from tp import maybe_init_dist
import argparse
from quantize import WeightOnlyInt8QuantHandler
from quantize import WeightOnlyInt4QuantHandler
from tp import apply_tp
import contextlib

def device_sync(device):
    if 'cuda' in device:
        torch.cuda.synchronize(device)
    elif 'cpu' in device or 'mps' in device:
        pass
    else:
        print(f'device={device} is not yet supported')
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch._functorch.config.enable_autograd_cache = True
default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
create_block_mask = torch.compile(create_block_mask)
wd = Path('/mnt/autor_name/haoTingDeWenJianJia/gpt-fast/generate.py').parent.parent.resolve()
sys.path.append(str(wd))
from model import Transformer
from tokenizer import get_tokenizer

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, block_mask: BlockMask, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    assert input_pos.shape[-1] == 1
    block_index = input_pos // block_mask.BLOCK_SIZE[0]
    mask = block_mask[:, :, block_index]
    mask.mask_mod = block_mask.mask_mod
    mask.seq_lengths = (1, model.max_seq_length)
    logits = model(mask, x, input_pos)
    return sample(logits, **sampling_kwargs)

def model_forward(model, x, input_pos):
    return model(x, input_pos)

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        model = Transformer.from_name(checkpoint_path.parent.name)
    if 'int8' in str(checkpoint_path):
        print('Using int8 weight-only quantization!')
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()
    if 'int4' in str(checkpoint_path):
        print('Using int4 weight-only quantization!')
        path_comps = checkpoint_path.name.split('.')
        groupsize = int(path_comps[-2][1:])
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()
    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if 'model' in checkpoint and 'stories' in str(checkpoint_path):
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint, assign=True)
    if use_tp:
        print('Applying tensor parallel to model ...')
        apply_tp(model)
    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for (name, child) in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum([p.numel() * p.dtype.itemsize for p in itertools.chain(child.parameters(), child.buffers())])
            params += sum([p.numel() for p in itertools.chain(child.parameters(), child.buffers())])
    return (model_size, params)
(B_INST, E_INST) = ('[INST]', '[/INST]')

def main(prompt: Union[int, str]='Hello, my name is', interactive: bool=False, num_samples: int=5, max_new_tokens: int=100, batch_size: int=1, top_k: int=200, temperature: float=0.8, checkpoint_path: Path=Path('checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth'), compile: bool=True, compile_prefill: bool=False, profile: Optional[Path]=None, draft_checkpoint_path: Optional[Path]=None, speculate_k: int=5, device=default_device) -> None:
    assert checkpoint_path.is_file(), checkpoint_path
    tokenizer_path = checkpoint_path.parent / 'tokenizer.model'
    assert tokenizer_path.is_file(), str(tokenizer_path)
    global print
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            print = lambda *args, **kwargs: None
    print(f'Using device={device}')
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None
    is_chat = 'chat' in str(checkpoint_path)
    print('Loading model ...')
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)
    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None
    device_sync(device=device)
    print(f'Time to load model: {time.time() - t0:.02f} seconds')
    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
    if isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        encoded = torch.randint(0, 1024, (prompt,), device=device, dtype=torch.int64)
    prompt_length = encoded.size(-1)
    torch.manual_seed(1234)
    (model_size, params) = _get_model_size(model)
    if compile:
        if is_speculative and use_tp:
            torch._inductor.config.triton.cudagraph_trees = False
        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode='reduce-overhead', fullgraph=True)
        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode='reduce-overhead', fullgraph=True)
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
    aggregate_metrics = {'tokens_per_sec': [], 'accept_counts': []}
    start = -1 if compile else 0
    for i in range(start, num_samples):
        device_sync(device=device)
        if i >= 0 and interactive:
            prompt = 'What is your prompt? '
            if is_chat:
                prompt = f'{B_INST} {prompt.strip()} {E_INST}'
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False

            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x: x
        t0 = time.perf_counter()
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
        with prof:
            (y, metrics) = exe.run('generate', model=model, prompt=encoded, max_new_tokens=max_new_tokens, batch_size=batch_size, draft_model=draft_model, speculate_k=speculate_k, interactive=interactive, callback=callback, temperature=temperature, top_k=top_k)
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        if i == -1:
            print(f'Compilation time: {time.perf_counter() - t0:.2f} seconds')
            continue
        if hasattr(prof, 'export_chrome_trace'):
            if use_tp:
                prof.export_chrome_trace(f'{profile}_rank_{rank}.json')
            else:
                prof.export_chrome_trace(f'{profile}.json')
        device_sync(device=device)
        t = time.perf_counter() - t0
        if not interactive:
            if batch_size > 1:
                print('Only displaying the first generation of the batch')
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        print(f'Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec')
        print(f'Bandwidth achieved: {model_size * generated_tokens_sec / 1000000000.0:.02f} GB/s')
        total_tokens_sec = y.numel() / t
        print(f'FLOPS achieved: {params * total_tokens_sec * 2 / 1000000000000.0:.02f} TF/s')
        print()
    print('==========')
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i / sum(counts_aggregated) for i in counts_aggregated]
        print(f'Acceptance probs: {acceptance_probs}')
        print(f'Mean Accepted: {sum([idx * i for (idx, i) in enumerate(counts_aggregated)]) / sum(counts_aggregated)}')
    print(f'Batch Size: {batch_size}')
    print(f'Prompt Length: {prompt_length}')
    print(f'Generated tokens: {max_new_tokens}')
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f'Memory used: {torch.cuda.max_memory_reserved() / 1000000000.0:.02f} GB')
main(prompt='Hello, my name is', interactive=False, num_samples=5, max_new_tokens=100, batch_size=1, top_k=200, temperature=0.8, checkpoint_path=Path('/mnt/autor_name/haoTingDeWenJianJia/gpt-fast/checkpoints/codellama/CodeLlama-7b-Python-hf/model_int8.pth'), compile=True, compile_prefill=False, profile=None, draft_checkpoint_path=None, speculate_k=5, device=default_device)