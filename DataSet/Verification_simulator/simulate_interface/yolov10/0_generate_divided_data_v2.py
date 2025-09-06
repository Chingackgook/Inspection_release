from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.yolov10 import ENV_DIR
from Inspection.adapters.custom_adapters.yolov10 import *
exe = Executor('yolov10', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/yolov10/ultralytics/data/converter.py'
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import LOGGER
from ultralytics.utils import TQDM
from ultralytics.utils.files import increment_path
from ultralytics.data import YOLODataset
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils import LOGGER
from ultralytics import SAM
from tqdm import tqdm
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.files import increment_path

def yolo_bbox2segment(im_dir, save_dir=None, sam_model='sam_b.pt'):
    """
    Converts existing object detection dataset (bounding boxes) to segmentation dataset or oriented bounding box (OBB)
    in YOLO format. Generates segmentation data using SAM auto-annotator as needed.

    Args:
        im_dir (str | Path): Path to image directory to convert.
        save_dir (str | Path): Path to save the generated labels, labels will be saved
            into `labels-segment` in the same directory level of `im_dir` if save_dir is None. Default: None.
        sam_model (str): Segmentation model to use for intermediate segmentation data; optional.

    Notes:
        The input directory structure assumed for dataset:

            - im_dir
                ├─ 001.jpg
                ├─ ..
                └─ NNN.jpg
            - labels
                ├─ 001.txt
                ├─ ..
                └─ NNN.txt
    """
    from ultralytics.data import YOLODataset
    from ultralytics.utils.ops import xywh2xyxy
    dataset = YOLODataset(im_dir, data=dict(names=list(range(1000))))
    if len(dataset.labels[0]['segments']) > 0:
        exe.run('info', detailed=False, verbose=True)
        return
    exe.run('info', detailed=False, verbose=True)
    sam_model = exe.create_interface_objects(interface_class_name='SAM', model='sam_b.pt')
    for l in tqdm(dataset.labels, total=len(dataset.labels), desc='Generating segment labels'):
        h, w = l['shape']
        boxes = l['bboxes']
        if len(boxes) == 0:
            continue
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        im = cv2.imread(l['im_file'])
        sam_results = exe.run('predict', source=im, bboxes=xywh2xyxy(boxes), stream=False)
        l['segments'] = sam_results[0].masks.xyn
    save_dir = Path(save_dir) if save_dir else Path(im_dir).parent / 'labels-segment'
    save_dir.mkdir(parents=True, exist_ok=True)
    for l in dataset.labels:
        texts = []
        lb_name = Path(l['im_file']).with_suffix('.txt').name
        txt_file = Path(FILE_RECORD_PATH) / lb_name
        cls = l['cls']
        for i, s in enumerate(l['segments']):
            line = (int(cls[i]), *s.reshape(-1))
            texts.append(('%g ' * len(line)).rstrip() % line)
        if texts:
            with open(txt_file, 'a') as f:
                f.writelines((text + '\n' for text in texts))
    exe.run('info', detailed=False, verbose=True)
# origin code:
#im_dir = RESOURCES_PATH + 'images/test_images_floder'
# add
im_dir = ENV_DIR
# end add
save_dir = 'path/to/save/labels'
sam_model = 'sam_b.pt'
yolo_bbox2segment(im_dir, save_dir, sam_model)