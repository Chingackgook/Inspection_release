from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.pycorrector import *
import sys
exe = Executor('pycorrector', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = (
    '/mnt/autor_name/haoTingDeWenJianJia/pycorrector/examples/kenlm/en_correct_demo.py'
    )
from pycorrector import EnSpellCorrector
sent = 'what happending? how to speling it, can you gorrect it?'
m = exe.create_interface_objects(interface_class_name='EnSpellCorrector')
details = exe.run('correct', sentence=sent)
print(details)
print()
sent_lst = ['what hapenning?', 'how to speling it', 'gorrect', 'i know']
for sent in sent_lst:
    details = exe.run('correct', sentence=sent)
    print('[error] ', details)
print()
sent = 'what is your name? shylock?'
r = exe.run('correct', sentence=sent)
print(r)
print('-' * 42)
my_dict = {'your': 120, 'name': 2, 'is': 1, 'shylock': 1, 'what': 1}
spell = exe.create_interface_objects(interface_class_name=
    'EnSpellCorrector', word_freq_dict=my_dict)
r = exe.run('correct', sentence=sent)
print(r)
print()
spell = exe.create_interface_objects(interface_class_name='EnSpellCorrector')
sent = 'what happt ? who is shylock.'
r = exe.run('correct', sentence=sent)
print(r)
print('-' * 42)
exe.run('set_en_custom_confusion_dict')
r = exe.run('correct', sentence=sent)
