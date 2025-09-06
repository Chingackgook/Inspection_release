from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.OpenChatKit import *
import os
import sys
import cmd
import torch
import conversation as convo
import retrieval.wikipedia as wp
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, StoppingCriteria, StoppingCriteriaList
from accelerate import infer_auto_device_map, init_empty_weights
exe = Executor('OpenChatKit', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
INFERENCE_DIR = os.path.dirname(os.path.abspath('/mnt/autor_name/haoTingDeWenJianJia/OpenChatKit/inference/bot.py'))
sys.path.append(os.path.join(INFERENCE_DIR, '..'))

class OpenChatKitShell(cmd.Cmd):
    intro = 'Welcome to OpenChatKit shell.   Type /help or /? to list commands.\n'
    prompt = '>>> '

    def __init__(self, gpu_id, model_name_or_path, max_tokens, sample, temperature, top_k, retrieval, max_memory, do_stream):
        super().__init__()
        self._gpu_id = gpu_id
        self._model_name_or_path = model_name_or_path
        self._max_tokens = max_tokens
        self._sample = sample
        self._temperature = temperature
        self._top_k = top_k
        self._retrieval = retrieval
        self._max_memory = max_memory
        self._do_stream = do_stream

    def preloop(self):
        print(f'Loading {self._model_name_or_path} to cuda:{self._gpu_id}...')
        self._model = exe.create_interface_objects(interface_class_name='ChatModel', model_name=self._model_name_or_path, gpu_id=self._gpu_id, max_memory=self._max_memory)
        if self._retrieval:
            print(f'Loading retrieval index...')
            self._index = wp.WikipediaIndex()
        self._convo = convo.Conversation(self._model.human_id, self._model.bot_id)

    def precmd(self, line):
        if line.startswith('/'):
            return line[1:]
        else:
            return 'say ' + line

    def do_say(self, arg):
        if self._retrieval:
            results = self._index.search(arg)
            if len(results) > 0:
                self._convo.push_context_turn(results[0])
        self._convo.push_human_turn(arg)
        output = exe.run('do_inference', prompt=self._convo.get_raw_prompt(), max_new_tokens=self._max_tokens, do_sample=self._sample, temperature=self._temperature, top_k=self._top_k, stream_callback=lambda x: print(x, end='', flush=True) if self._do_stream else None)
        self._convo.push_model_response(output)
        print('' if self._do_stream else self._convo.get_last_turn())

    def do_raw_say(self, arg):
        output = self._model.do_inference(arg, self._max_tokens, self._sample, self._temperature, self._top_k)
        print(output)

    def do_raw_prompt(self, arg):
        print(self._convo.get_raw_prompt())

    def do_reset(self, arg):
        self._convo = convo.Conversation(self._model.human_id, self._model.bot_id)

    def do_hyperparameters(self, arg):
        print(f'Hyperparameters:\n  max_tokens: {self._max_tokens}\n  sample: {self._sample}\n  temperature: {self._temperature}\n  top_k: {self._top_k}')

    def do_quit(self, arg):
        return True
gpu_id = 0
model_name_or_path = f'{INFERENCE_DIR}/../huggingface_models/Pythia-Chat-Base-7B'
max_tokens = 128
sample = True
temperature = 0.6
top_k = 40
retrieval = False
max_memory = None
do_stream = True
shell = OpenChatKitShell(gpu_id, model_name_or_path, max_tokens, sample, temperature, top_k, retrieval, max_memory, do_stream)
shell.preloop()
shell.do_say('Hello, how are you?')