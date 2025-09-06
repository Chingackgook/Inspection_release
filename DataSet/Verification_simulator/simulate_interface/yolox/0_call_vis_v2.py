from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.yolox import *
exe = Executor('yolox', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/YOLOX/demo/ONNXRuntime/onnx_inference.py'
import argparse
import os
import cv2
import numpy as np
import onnxruntime
from yolox.data.data_augment import preproc as preprocess
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import mkdir, multiclass_nms, demo_postprocess

def run_detection():
    model_path = 'yolox.onnx'
    image_path = 'test_image.png'
    output_dir = 'demo_output'
    score_thr = 0.3
    input_shape = '640,640'
    input_shape = tuple(map(int, input_shape.split(',')))
    origin_img = cv2.imread(image_path)
    (img, ratio) = preprocess(origin_img, input_shape)
    session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        (final_boxes, final_scores, final_cls_inds) = (dets[:, :4], dets[:, 4], dets[:, 5])
        origin_img = exe.run('vis', img=origin_img, boxes=final_boxes, scores=final_scores, cls_ids=final_cls_inds, conf=score_thr, class_names=COCO_CLASSES)
    mkdir(output_dir)
    output_path = os.path.join(FILE_RECORD_PATH, os.path.basename(image_path))
    cv2.imwrite(output_path, origin_img)
run_detection()