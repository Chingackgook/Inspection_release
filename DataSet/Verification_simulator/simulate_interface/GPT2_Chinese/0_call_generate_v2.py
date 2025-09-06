from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.GPT2_Chinese import *
exe = Executor('GPT2_Chinese', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/GPT2-Chinese/generate.py'
import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
from tokenizations import tokenization_bert_word_level as tokenization_bert
from tokenizations import tokenization_bert


def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    cp = ord(char)
    if (cp >= 19968 and cp <= 40959 or cp >= 13312 and cp <= 19903 or cp >=
        131072 and cp <= 173791 or cp >= 173824 and cp <= 177983 or cp >= 
        177984 and cp <= 178207 or cp >= 178208 and cp <= 183983 or cp >= 
        63744 and cp <= 64255 or cp >= 194560 and cp <= 195103):
        return True
    return False


def main():
    args = {'device': '0,1,2,3', 'length': -1, 'batch_size': 1, 'nsamples':
        10, 'temperature': 1.0, 'topk': 8, 'topp': 0.0, 'model_config':
        'config/model_config_small.json', 'tokenizer_path':
        'cache/vocab_small.txt', 'model_path': 'model/final_model',
        'prefix': '萧炎', 'no_wordpiece': False, 'segment': False,
        'fast_pattern': False, 'save_samples': True, 'save_samples_path':
        FILE_RECORD_PATH, 'repetition_penalty': 1.0}
    print('args:\n' + str(args))
    if args['segment']:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert
    os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
    length = args['length']
    batch_size = args['batch_size']
    nsamples = args['nsamples']
    temperature = args['temperature']
    topk = args['topk']
    topp = args['topp']
    repetition_penalty = args['repetition_penalty']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args[
        'tokenizer_path'])
    model = GPT2LMHeadModel.from_pretrained(args['model_path'])
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    if length == -1:
        length = model.config.n_ctx
    if args['save_samples']:
        if not os.path.exists(args['save_samples_path']):
            os.makedirs(args['save_samples_path'])
        samples_file = open(args['save_samples_path'] + '/samples.txt', 'w',
            encoding='utf8')
    generated = 0
    while generated < nsamples:
        raw_text = args['prefix']
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize
            (raw_text))
        for _ in range(nsamples // batch_size):
            out = exe.run('generate', n_ctx=n_ctx, model=model, context=
                context_tokens, length=length, is_fast_pattern=args[
                'fast_pattern'], tokenizer=tokenizer, temperature=
                temperature, top_k=topk, top_p=topp, repitition_penalty=
                repetition_penalty, device=device)
            for i in range(batch_size):
                generated += 1
                text = tokenizer.convert_ids_to_tokens(out)
                for i, item in enumerate(text[:-1]):
                    if is_word(item) and is_word(text[i + 1]):
                        text[i] = item + ' '
                for i, item in enumerate(text):
                    if item == '[MASK]':
                        text[i] = ''
                    elif item == '[CLS]':
                        text[i] = '\n\n'
                    elif item == '[SEP]':
                        text[i] = '\n'
                info = '=' * 40 + ' SAMPLE ' + str(generated
                    ) + ' ' + '=' * 40 + '\n'
                print(info)
                text = ''.join(text).replace('##', '').strip()
                print(text)
                if args['save_samples']:
                    samples_file.write(info)
                    samples_file.write(text)
                    samples_file.write('\n')
                    samples_file.write('=' * 90)
                    samples_file.write('\n' * 2)
        print('=' * 80)
    if args['save_samples']:
        samples_file.close()


main()
