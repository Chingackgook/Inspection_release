from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.MonkeyOCR import *
exe = Executor('MonkeyOCR', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/MonkeyOCR/parse.py'
import os
import time
import argparse
import sys
import torch.distributed as dist
from magic_pdf.utils.load_image import pdf_to_images
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.data.dataset import ImageDataset
from magic_pdf.data.dataset import MultiFileDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
from magic_pdf.model.custom_model import MonkeyOCR
TASK_INSTRUCTIONS = {'text': 'Please output the text content from the image.', 'formula': 'Please write out the expression of the formula in the image using LaTeX format.', 'table': 'This is the image of a table. Please output the table in html format.'}

def main():
    input_path = RESOURCES_PATH + 'images/test_image.pdf'
    output_dir = FILE_RECORD_PATH
    config_path = 'model_configs.yaml'
    task = 'text'
    split_pages = False
    group_size = None
    pred_abandon = False
    MonkeyOCR_model = None
    try:
        if os.path.isdir(input_path):
            result_dir = exe.run('parse_folder', folder_path=input_path, output_dir=output_dir, config_path=config_path, task=task, split_pages=split_pages, group_size=group_size, pred_abandon=pred_abandon)
            if task:
                if group_size:
                    print(f'\n✅ Folder processing with single task ({task}) recognition and image grouping (size: {group_size}) completed! Results saved in: {result_dir}')
                else:
                    print(f'\n✅ Folder processing with single task ({task}) recognition completed! Results saved in: {result_dir}')
            elif group_size:
                print(f'\n✅ Folder processing with image grouping (size: {group_size}) completed! Results saved in: {result_dir}')
            else:
                print(f'\n✅ Folder processing completed! Results saved in: {result_dir}')
        elif os.path.isfile(input_path):
            print('Loading model...')
            MonkeyOCR_model = MonkeyOCR(config_path)
            if task:
                result_dir = exe.run('single_task_recognition', input_file=input_path, output_dir=output_dir, MonkeyOCR_model=MonkeyOCR_model, task=task)
                print(f'\n✅ Single task ({task}) recognition completed! Results saved in: {result_dir}')
            else:
                result_dir = exe.run('parse_file', input_file=input_path, output_dir=output_dir, MonkeyOCR_model=MonkeyOCR_model, split_pages=split_pages, pred_abandon=pred_abandon)
                print(f'\n✅ Parsing completed! Results saved in: {result_dir}')
        else:
            raise FileNotFoundError(f'Input path does not exist: {input_path}')
    except Exception as e:
        print(f'\n❌ Processing failed: {str(e)}', file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            if MonkeyOCR_model is not None:
                if hasattr(MonkeyOCR_model, 'chat_model') and hasattr(MonkeyOCR_model.chat_model, 'close'):
                    MonkeyOCR_model.chat_model.close()
            time.sleep(1.0)
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as cleanup_error:
            print(f'Warning: Error during final cleanup: {cleanup_error}')
main()