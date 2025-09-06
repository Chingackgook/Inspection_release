# pycorrector 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'pycorrector/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/pycorrector')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/pycorrector')

# you can add your custom imports here
from pycorrector.en_spell_corrector import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        try:
            if interface_class_name == 'EnSpellCorrector':
                # Create interface object
                self.en_spell_corrector = EnSpellCorrector(**kwargs)
                self.result.interface_return = self.en_spell_corrector
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.en_spell_corrector = EnSpellCorrector(**kwargs)
                self.result.interface_return = self.en_spell_corrector

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == 'correct':
                self.result.interface_return = self.en_spell_corrector.correct(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'correct_batch':
                self.result.interface_return = self.en_spell_corrector.correct_batch(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'set_en_custom_confusion_dict':
                self.result.interface_return = self.en_spell_corrector.set_en_custom_confusion_dict(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'correct_word':
                self.result.interface_return = self.en_spell_corrector.correct_word(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'candidates':
                self.result.interface_return = self.en_spell_corrector.candidates(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'probability':
                self.result.interface_return = self.en_spell_corrector.probability(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'known':
                self.result.interface_return = self.en_spell_corrector.known(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'edits1':
                self.result.interface_return = self.en_spell_corrector.edits1(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'edits2':
                self.result.interface_return = self.en_spell_corrector.edits2(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
