import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.deepface import ENV_DIR
from Inspection.adapters.custom_adapters.deepface import *
exe = Executor('deepface', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# built-in dependencies
import traceback
from typing import Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

# 假设我们需要加载模型
model_name = "VGG-Face"
detector_backend = "opencv"
distance_metric = "cosine"
enforce_detection = True
align = True
anti_spoofing = False

# 加载模型
exe.create_interface_objects(model_name=model_name)

# pylint: disable=broad-except

def represent(
    img_path: Union[str, np.ndarray],
    max_faces: Optional[int] = None,
):
    try:
        result = {}
        embedding_objs = exe.run("represent", img_path=img_path, model_name=model_name, 
                                  detector_backend=detector_backend, 
                                  enforce_detection=enforce_detection, 
                                  align=align, anti_spoofing=anti_spoofing, 
                                  max_faces=max_faces)
        result["results"] = embedding_objs
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
):
    try:
        obj = exe.run("verify", img1_path=img1_path, img2_path=img2_path, 
                       model_name=model_name, detector_backend=detector_backend, 
                       distance_metric=distance_metric, align=align, 
                       enforce_detection=enforce_detection, 
                       anti_spoofing=anti_spoofing)
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, np.ndarray],
    actions: list = ["emotion", "age", "gender", "race"],
):
    try:
        result = {}
        demographies = exe.run("analyze", img_path=img_path, actions=actions, 
                               detector_backend=detector_backend, 
                               enforce_detection=enforce_detection, 
                               align=align, anti_spoofing=anti_spoofing)
        result["results"] = demographies
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def find(
    img_path: Union[str, np.ndarray],
    db_path: str,
):
    try:
        result = exe.run("find", img_path=img_path, db_path=db_path, 
                         model_name=model_name, distance_metric=distance_metric, 
                         enforce_detection=enforce_detection, 
                         detector_backend=detector_backend, align=align, 
                         anti_spoofing=anti_spoofing)
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while finding: {str(err)} - {tb_str}"}, 400


def stream(
    db_path: str,
    source: Optional[Union[int, str]] = 0,
    time_threshold: int = 5,
    frame_threshold: int = 5,
    output_path: Optional[str] = None,
    debug: bool = False,
):
    try:
        exe.run("stream", db_path=db_path, model_name=model_name, 
                detector_backend=detector_backend, distance_metric=distance_metric, 
                enable_face_analysis=True, source=source, 
                time_threshold=time_threshold, frame_threshold=frame_threshold, 
                anti_spoofing=anti_spoofing, output_path=FILE_RECORD_PATH if output_path is None else output_path, 
                debug=debug)
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)


def extract_faces(
    img_path: Union[str, np.ndarray],
    expand_percentage: int = 0,
    grayscale: bool = False,
    color_face: str = "rgb",
    normalize_face: bool = True,
):
    try:
        result = exe.run("extract_faces", img_path=img_path, 
                         detector_backend=detector_backend, 
                         enforce_detection=enforce_detection, 
                         align=align, expand_percentage=expand_percentage, 
                         grayscale=grayscale, color_face=color_face, 
                         normalize_face=normalize_face, 
                         anti_spoofing=anti_spoofing)
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while extracting faces: {str(err)} - {tb_str}"}, 400


# 直接运行主逻辑
#cli()
# 示例1：人脸验证 - 比较两张图片是否为同一人
img1_path = ENV_DIR+"deepface/img1.jpg"  # 替换为你的第一张图片路径
img2_path = ENV_DIR+"deepface/img2.jpg"  # 替换为你的第二张图片路径

verification = verify(img1_path, img2_path)
if "error" not in verification:
    print(f"验证结果: {'匹配' if verification['verified'] else '不匹配'}")
    print(f"相似度得分: {verification['distance']:.4f}")
    print(f"使用的算法: {verification['model']}")
else:
    print("验证失败:", verification["error"])