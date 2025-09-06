from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.surya import *
exe = Executor('surya', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/surya/surya/scripts/ocr_text.py'
import os
import click
import json
import time
from collections import defaultdict
from surya.common.surya.schema import TaskNames
from surya.detection import DetectionPredictor
from surya.debug.text import draw_text_on_image
from surya.logging import configure_logging, get_logger
from surya.recognition import RecognitionPredictor
from surya.scripts.config import CLILoader
configure_logging()
logger = get_logger()
input_path = RESOURCES_PATH + 'images/test_images_floder'
task_name = TaskNames.ocr_with_boxes
disable_math = False

def ocr_text_cli(input_path: str, task_name: str, disable_math: bool):
    # add
    cli_options = {
        "images": True,
        "debug": False,
        "output_dir": FILE_RECORD_PATH+ '/surya',
    }
    # end add
    loader = CLILoader(input_path, cli_options, highres=True)
    task_names = [task_name] * len(loader.images)
    det_predictor = DetectionPredictor()
    rec_predictor = exe.create_interface_objects(interface_class_name='RecognitionPredictor', checkpoint=None, device=settings.TORCH_DEVICE_MODEL, dtype=None)
    start = time.time()
    predictions_by_image = exe.run('__call__', images=loader.images, task_names=task_names, det_predictor=det_predictor, highres_images=loader.highres_images, math_mode=not disable_math)
    if loader.debug:
        logger.debug(f'OCR took {time.time() - start:.2f} seconds')
        max_chars = max([len(line.text) for p in predictions_by_image for line in p.text_lines])
        logger.debug(f'Max chars: {max_chars}')
    if loader.save_images:
        for idx, (name, image, pred) in enumerate(zip(loader.names, loader.images, predictions_by_image)):
            bboxes = [line.bbox for line in pred.text_lines]
            pred_text = [line.text for line in pred.text_lines]
            page_image = draw_text_on_image(bboxes, pred_text, image.size)
            page_image.save(os.path.join(FILE_RECORD_PATH, f'{name}_{idx}_text.png'))
    out_preds = defaultdict(list)
    for name, pred, image in zip(loader.names, predictions_by_image, loader.images):
        out_pred = pred.model_dump()
        out_pred['page'] = len(out_preds[name]) + 1
        out_preds[name].append(out_pred)
    with open(os.path.join(FILE_RECORD_PATH, 'results.json'), 'w+', encoding='utf-8') as f:
        json.dump(out_preds, f, ensure_ascii=False)
    logger.info(f'Wrote results to {FILE_RECORD_PATH}')
ocr_text_cli(input_path, task_name, disable_math)