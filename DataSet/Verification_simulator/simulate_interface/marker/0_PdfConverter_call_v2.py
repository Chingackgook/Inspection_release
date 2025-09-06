from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.marker import *
exe = Executor('marker', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/marker/benchmarks/throughput/main.py'
import time
import torch
import click
import pypdfium2 as pdfium
from tqdm import tqdm
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
pdf_path = RESOURCES_PATH + 'images/test_image.pdf'

def run_conversion(pdf_path):
    print(f'Converting {pdf_path} to markdown...')
    pdf = pdfium.PdfDocument(pdf_path)
    page_count = len(pdf)
    pdf.close()
    model_dict = create_model_dict()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in tqdm(range(10), desc='Benchmarking'):
        block_converter = exe.create_interface_objects(interface_class_name='PdfConverter', artifact_dict=model_dict, config={'disable_tqdm': True})
        start = time.time()
        output = exe.run('__call__', filepath=pdf_path)
        total = time.time() - start
        times.append(total)
    max_gpu_vram = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f'Converted {page_count} pages in {sum(times) / len(times):.2f} seconds.')
    print(f'Max GPU VRAM: {max_gpu_vram:.2f} GB')
run_conversion(pdf_path)