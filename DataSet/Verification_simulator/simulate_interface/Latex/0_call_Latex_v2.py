from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Latex import *
exe = Executor('Latex', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import sys
from http import HTTPStatus
from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image
from io import BytesIO
from pix2tex.cli import LatexOCR
model = None

def load_model():
    global model
    if model is None:
        model = exe.create_interface_objects(interface_class_name='LatexOCR')

def predict(file_path: str) -> str:
    """"""
    global model
    image = Image.open(file_path)
    return exe.run('call', img=image)
load_model()
test_image_path = RESOURCES_PATH + 'images/test_image.png'
predicted_latex = predict(test_image_path)
print(predicted_latex)