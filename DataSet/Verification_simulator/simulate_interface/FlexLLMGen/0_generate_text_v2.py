from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.FlexLLMGen import *
import sys
import argparse
from flexllmgen.flex_opt import Policy
from flexllmgen.flex_opt import OptLM
from flexllmgen.flex_opt import ExecutionEnv
from flexllmgen.flex_opt import CompressionConfig
from flexllmgen.flex_opt import str2bool
from transformers import AutoTokenizer
exe = Executor('FlexLLMGen', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/FlexLLMGen/flexllmgen/apps/completion.py'
'Complete sentences with FlexLLMGen and OPT models.'

def run_model():
    model_name = 'facebook/opt-1.3b'
    path_to_weights = '~/opt_weights'
    offload_dir = '~/flexllmgen_offload_dir'
    percent = [100, 0, 100, 0, 100, 0]
    pin_weight = True
    cpu_cache_compute = False
    compress_weight = False
    compress_cache = False
    prompts = ['Question: Where were the 2004 Olympics held?\nAnswer: Athens, Greece\nQuestion: What is the longest river on the earth?\nAnswer:', 'Extract the airport codes from this text.\nText: "I want a flight from New York to San Francisco."\nAirport codes: JFK, SFO.\nText: "I want you to book a flight from Phoenix to Las Vegas."\nAirport codes:']
    env = ExecutionEnv.create(offload_dir)
    policy = Policy(len(prompts), 1, percent[0], percent[1], percent[2], percent[3], percent[4], percent[5], overlap=True, sep_layer=True, pin_weight=pin_weight, cpu_cache_compute=cpu_cache_compute, attn_sparsity=1.0, compress_weight=compress_weight, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False), compress_cache=compress_cache, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False))
    print('Initialize...')
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.add_bos_token = False
    stop = tokenizer('\n').input_ids[0]
    model = exe.create_interface_objects(interface_class_name='OptLM', config=model_name, env=env, path=path_to_weights, policy=policy)
    print('Generate...')
    inputs = tokenizer(prompts, padding='max_length', max_length=128)
    output_ids = exe.run('generate', inputs=inputs.input_ids, do_sample=True, temperature=0.7, max_new_tokens=32, stop=stop)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print('Outputs:\n' + 70 * '-')
    for i in [0, len(outputs) - 1]:
        print(f'{i}: {outputs[i]}')
        print('-' * 70)
    print('Shutdown...')
    env.close_copy_threads()
run_model()