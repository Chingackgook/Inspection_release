import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.insightface import ENV_DIR
from Inspection.adapters.custom_adapters.insightface import *
exe = Executor('insightface', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import datetime
import numpy as np
import os
import os.path as osp
import glob
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

assert insightface.__version__ >= '0.7'


# 使用 exe.create_interface_objects 加载模型
exe.create_interface_objects(name='buffalo_sc')

# 替换 prepare 方法
exe.run("prepare", ctx_id=0, det_size=(640, 640))

# https://objects.githubusercontent.com/github-production-release-asset-2e65be/102057483/9517799d-b91c-48b3-a267-258a94383dfd?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=releaseassetproduction%2F20250428%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250428T094336Z&X-Amz-Expires=300&X-Amz-Signature=8425209ac8d01feb28afcb2358c14ef18b7b731e26ece07b84f50e241a9024ee&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3Dbuffalo_s.zip&response-content-type=application%2Foctet-stream
swapper = insightface.model_zoo.get_model('buffalo_sc', download=True, download_zip=True)

# 获取图像
img = ins_get_image('t1')

# 替换 get 方法
faces = exe.run("get", img=img, max_num=0, det_metric='default')

faces = sorted(faces, key=lambda x: x.bbox[0])
assert len(faces) == 6
source_face = faces[2]
res = img.copy()

for face in faces:
    res = swapper.get(res, face, source_face, paste_back=True)

cv2.imwrite(osp.join(FILE_RECORD_PATH, "t1_swapped.jpg"), res)

res = []
for face in faces:
    _img, _ = swapper.get(img, face, source_face, paste_back=False)
    res.append(_img)

res = np.concatenate(res, axis=1)
cv2.imwrite(osp.join(FILE_RECORD_PATH, "t1_swapped2.jpg"), res)
