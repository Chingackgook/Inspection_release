from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.moshi import *
exe = Executor('moshi', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/moshi/moshi/moshi/run_inference.py'
import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys
import time
import numpy as np
import sentencepiece
import torch
import sphn
from moshi.client_utils import log
from moshi.client_utils import AnyPrinter
from moshi.client_utils import Printer
from moshi.client_utils import RawPrinter
from moshi.conditioners import ConditionAttributes
from moshi.conditioners import ConditionTensors
from moshi.models import loaders
from moshi.models import MimiModel
from moshi.models import LMModel
from moshi.models import LMGen
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import sentencepiece
import torch
import sphn
from moshi.client_utils import log, AnyPrinter, Printer, RawPrinter
from moshi.conditioners import ConditionAttributes, ConditionTensors
from moshi.models import loaders, MimiModel, LMModel, LMGen

def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

def get_condition_tensors(model_type: str, lm: LMModel, batch_size: int, cfg_coef: float) -> ConditionTensors:
    condition_tensors = {}
    if lm.condition_provider is not None and lm.condition_provider.conditioners:
        conditions: list[ConditionAttributes] | None = None
        if model_type == 'hibiki':
            conditions = [ConditionAttributes(text={'description': 'very_good'}, tensor={}) for _ in range(batch_size)]
            if cfg_coef != 1.0:
                conditions += [ConditionAttributes(text={'description': 'very_bad'}, tensor={}) for _ in range(batch_size)]
        else:
            raise RuntimeError(f'Model expects conditioning but model type {model_type} is not supported.')
        assert conditions is not None
        prepared = lm.condition_provider.prepare(conditions)
        condition_tensors = lm.condition_provider(prepared)
    return condition_tensors

@dataclass
class InferenceState:
    mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen

    def __init__(self, checkpoint_info: loaders.CheckpointInfo, mimi: MimiModel, text_tokenizer: sentencepiece.SentencePieceProcessor, lm: LMModel, batch_size: int, cfg_coef: float, device: str | torch.device, **kwargs):
        self.checkpoint_info = checkpoint_info
        model_type = checkpoint_info.model_type
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size, cfg_coef)
        self.lm_gen = exe.create_interface_objects(interface_class_name='LMGen', lm_model=lm, use_sampling=True, temp=0.8, temp_text=0.7, top_k=250, top_k_text=25, cfg_coef=cfg_coef, check=False, condition_tensors=condition_tensors, on_text_hook=None, on_text_logits_hook=None, on_audio_hook=None, support_out_of_sync=False, cfg_is_masked_until=None, cfg_is_no_text=False)
        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.batch_size = batch_size
        self.mimi.streaming_forever(batch_size)
        self.lm_gen.streaming_forever(batch_size)
        self.printer: AnyPrinter
        if sys.stdout.isatty():
            self.printer = Printer()
        else:
            self.printer = RawPrinter()

    def run(self, in_pcms: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out_pcms_per_item: list[list[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        out_text_tokens_per_item: list[list[torch.Tensor]] = [[] for _ in range(self.batch_size)]
        eos_reached: list[bool] = [False] * self.batch_size
        need_eos_input: bool = True
        self.printer.log('info', f'starting inference, sampling: {self.lm_gen.use_sampling}, audio temp: {self.lm_gen.temp}, text temp: {self.lm_gen.temp_text}')
        device = self.lm_gen.lm_model.device
        start_time = time.time()
        ntokens = 0
        first_frame = True
        if self.model_type == 'stt':
            stt_config = self.checkpoint_info.stt_config
            pad_right = stt_config.get('audio_delay_seconds', 0.0)
            pad_left = stt_config.get('audio_silence_prefix_seconds', 0.0)
            pad_left = int(pad_left * 24000)
            pad_right = int((pad_right + 1.0) * 24000)
            in_pcms = torch.nn.functional.pad(in_pcms, (pad_left, pad_right), mode='constant')
        chunks = deque([chunk for chunk in in_pcms.split(self.frame_size, dim=2) if chunk.shape[-1] == self.frame_size])
        self.printer.print_header()
        while not all(eos_reached):
            if chunks:
                chunk = chunks.popleft()
                codes = self.mimi.encode(chunk)
            elif self.model_type == 'hibiki':
                if need_eos_input:
                    need_eos_input = False
                    eos_value = self.mimi.cardinality
                    codes = torch.full((self.batch_size, self.mimi.num_codebooks, 1), eos_value, device=device, dtype=torch.long)
                else:
                    silence = torch.zeros((self.batch_size, self.mimi.channels, self.frame_size), device=device)
                    codes = self.mimi.encode(silence)
            else:
                break
            if first_frame:
                tokens = exe.run('step', input_tokens=codes)
                if max(self.lm_gen.lm_model.delays) > 0:
                    assert tokens is None
                first_frame = False
            tokens = exe.run('step', input_tokens=codes)
            if tokens is None:
                continue
            assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
            if self.lm_gen.lm_model.dep_q > 0:
                out_pcm = self.mimi.decode(tokens[:, 1:]).cpu()
                for b, (one_text, one_pcm) in enumerate(zip(tokens[:, 0].cpu(), out_pcm)):
                    if eos_reached[b]:
                        continue
                    elif one_text.item() == self.text_tokenizer.eos_id():
                        if need_eos_input:
                            self.printer.log('warning', 'EOS sampled too early.')
                        else:
                            eos_reached[b] = True
                    out_text_tokens_per_item[b].append(one_text)
                    out_pcms_per_item[b].append(one_pcm)
                    if b == 0:
                        if one_text.item() not in [0, 3]:
                            text = self.text_tokenizer.id_to_piece(one_text.item())
                            text = text.replace('▁', ' ')
                            self.printer.print_token(text)
            else:
                one_text = tokens[0, 0].cpu()
                if one_text.item() not in [0, 3]:
                    text = self.text_tokenizer.id_to_piece(one_text.item())
                    text = text.replace('▁', ' ')
                    self.printer.print_token(text)
            ntokens += 1
        dt = time.time() - start_time
        self.printer.log('info', f'processed {ntokens} steps in {dt:.0f}s, {1000 * dt / ntokens:.2f}ms/step')
        if self.lm_gen.lm_model.dep_q > 0:
            out = [(torch.cat(one_texts, dim=0), torch.cat(one_pcms, dim=1)) for one_texts, one_pcms in zip(out_text_tokens_per_item, out_pcms_per_item)]
            return out
        else:
            return []
        

# add
tokenizer_path = None
moshi_weight_path = '/mnt/autor_name/haoTingDeWenJianJia/moshi/moshimodel/model.safetensors'
mimi_weight_path = None
config_path = None
# end add
# origin code:
# tokenizer_path = ''
# moshi_weight_path = ''
# mimi_weight_path = ''
# config_path = ''
hf_repo = loaders.DEFAULT_REPO
batch_size = 8
device = 'cuda'
dtype = torch.bfloat16
cfg_coef = 1.0
infile = RESOURCES_PATH + 'audios/test_audio.wav'
outfile = FILE_RECORD_PATH + '/output/audio.wav'
seed_all(4242)
log('info', 'retrieving checkpoint')
checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo, moshi_weight_path, mimi_weight_path, tokenizer_path, config_path)
log('info', 'loading mimi')
mimi = checkpoint_info.get_mimi(device=device)
log('info', 'mimi loaded')
text_tokenizer = checkpoint_info.get_text_tokenizer()
log('info', 'loading moshi')
lm = checkpoint_info.get_moshi(device=device, dtype=dtype)
log('info', 'moshi loaded')
if lm.dep_q == 0:
    batch_size = 1
log('info', f'loading input file {infile}')
in_pcms, _ = sphn.read(infile, sample_rate=mimi.sample_rate)
in_pcms = torch.from_numpy(in_pcms).to(device=device)
in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)
state = InferenceState(checkpoint_info, mimi, text_tokenizer, lm, batch_size, cfg_coef, device, **checkpoint_info.lm_gen_config)
out_items = state.run(in_pcms)
if outfile:
    outfile_path = Path(outfile)
    for index, (_, out_pcm) in enumerate(out_items):
        if len(out_items) > 1:
            outfile_ = outfile_path.with_name(f'{outfile_path.stem}-{index}{outfile_path.suffix}')
        else:
            outfile_ = outfile_path
        duration = out_pcm.shape[1] / mimi.sample_rate
        log('info', f'writing {outfile_} with duration {duration:.1f} sec.')
        # old sphn.write_wav(str(outfile_), out_pcm[0].numpy(), sample_rate=mimi.sample_rate)
        sphn.write_wav(str(outfile_), out_pcm[0].detach().numpy(), sample_rate=mimi.sample_rate)