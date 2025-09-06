from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.deepface import *
exe = Executor('deepface', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import traceback
from typing import Optional, Union
import numpy as np
from deepface import DeepFace
from deepface.commons.logger import Logger
logger = Logger()

def represent(img_path: Union[str, np.ndarray], model_name: str, detector_backend: str, enforce_detection: bool, align: bool, anti_spoofing: bool, max_faces: Optional[int]=None):
    try:
        result = {}
        # Using FILE_RECORD_PATH for output
        result['results'] = exe.run('represent', img_path=img_path, model_name=model_name, detector_backend=detector_backend, enforce_detection=enforce_detection, align=align, anti_spoofing=anti_spoofing, max_faces=max_faces)
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return ({'error': f'Exception while representing: {str(err)} - {tb_str}'}, 400)

def verify(img1_path: Union[str, np.ndarray], img2_path: Union[str, np.ndarray], model_name: str, detector_backend: str, distance_metric: str, enforce_detection: bool, align: bool, anti_spoofing: bool):
    try:
        # Using FILE_RECORD_PATH for output
        obj = exe.run('verify', img1_path=img1_path, img2_path=img2_path, model_name=model_name, detector_backend=detector_backend, distance_metric=distance_metric, align=align, enforce_detection=enforce_detection, anti_spoofing=anti_spoofing)
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return ({'error': f'Exception while verifying: {str(err)} - {tb_str}'}, 400)

def analyze(img_path: Union[str, np.ndarray], actions: list, detector_backend: str, enforce_detection: bool, align: bool, anti_spoofing: bool):
    try:
        result = {}
        # Using FILE_RECORD_PATH for output
        result['results'] = exe.run('analyze', img_path=img_path, actions=actions, detector_backend=detector_backend, enforce_detection=enforce_detection, align=align, anti_spoofing=anti_spoofing)
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return ({'error': f'Exception while analyzing: {str(err)} - {tb_str}'}, 400)

# Main logic starts here
exe.create_interface_objects(interface_class_name='DeepFace')

# Parts that may need manual modification:
img_path = RESOURCES_PATH + 'images/test_image.jpg'# Example usage of FILE_RECORD_PATH
model_name = 'VGG-Face'  # Parts that may need manual modification
detector_backend = 'opencv'  # Parts that may need manual modification
enforce_detection = True
align = True
anti_spoofing = False
max_faces = None

result = represent(img_path, model_name, detector_backend, enforce_detection, align, anti_spoofing, max_faces)
print(result)

img1_path = RESOURCES_PATH + 'images/test_image.jpg'# Example usage of FILE_RECORD_PATH
img2_path = RESOURCES_PATH + 'images/test_image.jpg'# Example usage of FILE_RECORD_PATH
distance_metric = 'cosine'  # Parts that may need manual modification
verification_result = verify(img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align, anti_spoofing)
print(verification_result)

actions = ['emotion', 'age', 'gender', 'race']  # Parts that may need manual modification
analysis_result = analyze(img_path, actions, detector_backend, enforce_detection, align, anti_spoofing)
print(analysis_result)