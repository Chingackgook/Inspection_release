from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.IOPaint import *
exe = Executor('IOPaint', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/IOPaint/iopaint/cli.py'
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import typer
from fastapi import FastAPI
from loguru import logger
from typer import Option
from typer_config import use_json_config
from iopaint.const import *
from iopaint.runtime import setup_model_dir
from iopaint.runtime import dump_environment_info
from iopaint.runtime import check_device
from iopaint.schema import InteractiveSegModel
from iopaint.schema import Device
from iopaint.schema import RealESRGANModel
from iopaint.schema import RemoveBGModel
from iopaint.download import cli_download_model
from iopaint.download import scan_models
from iopaint.batch_processing import batch_inpaint
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from iopaint.const import *
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.schema import InteractiveSegModel, Device, RealESRGANModel, RemoveBGModel
from iopaint.download import cli_download_model, scan_models
from iopaint.batch_processing import batch_inpaint
from loguru import logger
model = 'lama'
device = Device.cuda
image = Path(RESOURCES_PATH + 'images/test_images_floder')
mask = Path(RESOURCES_PATH + 'images/test_images_floder')
output = Path(FILE_RECORD_PATH)
config = None
concat = False
model_dir = DEFAULT_MODEL_DIR
scanned_models = scan_models()
if model not in [it.name for it in scanned_models]:
    logger.info(f'{model} not found in {model_dir}, trying to download')
    cli_download_model(model)
exe.run('batch_inpaint', model=model, device=device, image=image, mask=mask, output=output, config=config, concat=concat)