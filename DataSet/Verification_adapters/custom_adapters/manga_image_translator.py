# manga_image_translator 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'manga_image_translator/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/manga-image-translator')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/manga-image-translator')

# you can add your custom imports here
from manga_translator.mode.local import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Placeholder for MangaTranslatorLocal object

        try:
            if interface_class_name == 'MangaTranslatorLocal':
                # Create interface object for MangaTranslatorLocal
                self.class1_obj = MangaTranslatorLocal(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = MangaTranslatorLocal(**kwargs)
                self.result.interface_return = self.class1_obj

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
            if dispatch_key == 'safe_get_memory_info':
                self.result.interface_return = safe_get_memory_info()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'force_cleanup':
                force_cleanup()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'translate_path':
                self.result.interface_return = self.class1_obj.translate_path(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'translate_file':
                self.result.interface_return = self.class1_obj.translate_file(**kwargs)
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
