from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.RIFE import *
exe = Executor('RIFE', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/ECCV2022-RIFE/inference_img.py'
import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
from model.RIFE import Model
# Ins delete
# from model.RIFE_HDv2 import Model
# from model.RIFE_HD import Model
# from train_log.RIFE_HDv3 import Model
import os
import cv2
import torch
from torch.nn import functional as F
import warnings
warnings.filterwarnings('ignore')
img_paths = [RESOURCES_PATH + 'images/test_image.png', RESOURCES_PATH + 'images/test_image.png']
exp = 4
ratio = 0
rthreshold = 0.02
rmaxcycles = 8
modelDir = 'train_log'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
try:
    try:
        try:
            model = exe.create_interface_objects(interface_class_name='Model', local_rank=-1, arbitrary=False)
            var = exe.run('load_model', path=modelDir, rank=-1)
            print('Loaded v2.x HD model.')
        except:
            model = exe.create_interface_objects(interface_class_name='Model', local_rank=-1, arbitrary=False)
            var = exe.run('load_model', path=modelDir, rank=-1)
            print('Loaded v3.x HD model.')
    except:
        model = exe.create_interface_objects(interface_class_name='Model', local_rank=-1, arbitrary=False)
        var = exe.run('load_model', path=modelDir, rank=-1)
        print('Loaded v1.x HD model')
except:
    model = exe.create_interface_objects(interface_class_name='Model', local_rank=-1, arbitrary=False)
    var = exe.run('load_model', path=modelDir, rank=-1)
    print('Loaded ArXiv-RIFE model')
var = exe.run('eval')
if img_paths[0].endswith('.exr') and img_paths[1].endswith('.exr'):
    img0 = cv2.imread(img_paths[0], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img1 = cv2.imread(img_paths[1], cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    img0 = torch.tensor(img0.transpose(2, 0, 1)).to(device).unsqueeze(0)
    img1 = torch.tensor(img1.transpose(2, 0, 1)).to(device).unsqueeze(0)
else:
    img0 = cv2.imread(img_paths[0], cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(img_paths[1], cv2.IMREAD_UNCHANGED)
    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.0).unsqueeze(0)
(n, c, h, w) = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = F.pad(img0, padding)
img1 = F.pad(img1, padding)
if ratio:
    img_list = [img0]
    img0_ratio = 0.0
    img1_ratio = 1.0
    if ratio <= img0_ratio + rthreshold / 2:
        middle = img0
    elif ratio >= img1_ratio - rthreshold / 2:
        middle = img1
    else:
        tmp_img0 = img0
        tmp_img1 = img1
        for inference_cycle in range(rmaxcycles):
            middle = exe.run('inference', img0=tmp_img0, img1=tmp_img1)
            middle_ratio = (img0_ratio + img1_ratio) / 2
            if ratio - rthreshold / 2 <= middle_ratio <= ratio + rthreshold / 2:
                break
            if ratio > middle_ratio:
                tmp_img0 = middle
                img0_ratio = middle_ratio
            else:
                tmp_img1 = middle
                img1_ratio = middle_ratio
    img_list.append(middle)
    img_list.append(img1)
else:
    img_list = [img0, img1]
    for i in range(exp):
        tmp = []
        for j in range(len(img_list) - 1):
            mid = exe.run('inference', img0=img_list[j], img1=img_list[j + 1])
            tmp.append(img_list[j])
            tmp.append(mid)
        tmp.append(img1)
        img_list = tmp
if not os.path.exists('output'):
    os.mkdir('output')
for i in range(len(img_list)):
    if img_paths[0].endswith('.exr') and img_paths[1].endswith('.exr'):
        cv2.imwrite(os.path.join(FILE_RECORD_PATH, 'img{}.exr'.format(i)), img_list[i][0].cpu().numpy().transpose(1, 2, 0)[:h, :w], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF])
    else:
        cv2.imwrite(os.path.join(FILE_RECORD_PATH, 'img{}.png'.format(i)), (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])