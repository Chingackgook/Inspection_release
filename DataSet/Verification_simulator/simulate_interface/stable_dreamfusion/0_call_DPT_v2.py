from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.stable_dreamfusion import *
exe = Executor('stable_dreamfusion', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/stable-dreamfusion/preprocess_image.py'
import os
import sys
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from carvekit.api.high import HiInterface
from transformers import AutoProcessor
from transformers import Blip2ForConditionalGeneration
from dpt import DPTDepthModel
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

class BackgroundRemoval:

    def __init__(self, device='cuda'):
        from carvekit.api.high import HiInterface
        self.interface = HiInterface(object_type='object', batch_size_seg=5, batch_size_matting=1, device=device, seg_mask_size=640, matting_mask_size=2048, trimap_prob_threshold=231, trimap_dilation=30, trimap_erosion_iters=5, fp16=True)

    @torch.no_grad()
    def __call__(self, image):
        image = Image.fromarray(image)
        image = self.interface([image])[0]
        image = np.array(image)
        return image

class DPT:

    def __init__(self, task='depth', device='cuda'):
        self.task = task
        self.device = device
        from dpt import DPTDepthModel
        if task == 'depth':
            path = 'pretrained/omnidata/omnidata_dpt_depth_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384')
            self.aug = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])
        else:
            path = 'pretrained/omnidata/omnidata_dpt_normal_v2.ckpt'
            self.model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3)
            self.aug = transforms.Compose([transforms.Resize((384, 384)), transforms.ToTensor()])
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for (k, v) in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval().to(device)

    @torch.no_grad()
    def __call__(self, image):
        (H, W) = image.shape[:2]
        image = Image.fromarray(image)
        image = self.aug(image).unsqueeze(0).to(self.device)
        if self.task == 'depth':
            depth = self.model(image).clamp(0, 1)
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode='bicubic', align_corners=False)
            depth = depth.squeeze(1).cpu().numpy()
            return depth
        else:
            normal = self.model(image).clamp(0, 1)
            normal = F.interpolate(normal, size=(H, W), mode='bicubic', align_corners=False)
            normal = normal.cpu().numpy()
            return normal

def main():
    input_image_path = RESOURCES_PATH + 'images/test_image.png'
    output_size = 256
    border_ratio = 0.2
    recenter = True
    out_dir = os.path.dirname(input_image_path)
    out_rgba = os.path.join(FILE_RECORD_PATH, os.path.basename(input_image_path).split('.')[0] + '_rgba.png')
    out_depth = os.path.join(FILE_RECORD_PATH, os.path.basename(input_image_path).split('.')[0] + '_depth.png')
    out_normal = os.path.join(FILE_RECORD_PATH, os.path.basename(input_image_path).split('.')[0] + '_normal.png')
    print(f'[INFO] loading image...')
    image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f'[INFO] background removal...')
    carved_image = BackgroundRemoval()(image)
    mask = carved_image[..., -1] > 0
    print(f'[INFO] depth estimation...')
    dpt_depth_model = exe.create_interface_objects(interface_class_name='DPT', task='depth', device='cuda')
    depth = exe.run('__call__', image=image)[0]
    depth[mask] = (depth[mask] - depth[mask].min()) / (depth[mask].max() - depth[mask].min() + 1e-09)
    depth[~mask] = 0
    depth = (depth * 255).astype(np.uint8)
    del dpt_depth_model
    print(f'[INFO] normal estimation...')
    dpt_normal_model = exe.create_interface_objects(interface_class_name='DPT', task='normal', device='cuda')
    normal = exe.run('__call__', image=image)[0]
    normal = (normal * 255).astype(np.uint8).transpose(1, 2, 0)
    normal[~mask] = 0
    del dpt_normal_model
    if recenter:
        print(f'[INFO] recenter...')
        final_rgba = np.zeros((output_size, output_size, 4), dtype=np.uint8)
        final_depth = np.zeros((output_size, output_size), dtype=np.uint8)
        final_normal = np.zeros((output_size, output_size, 3), dtype=np.uint8)
        coords = np.nonzero(mask)
        (x_min, x_max) = (coords[0].min(), coords[0].max())
        (y_min, y_max) = (coords[1].min(), coords[1].max())
        h = x_max - x_min
        w = y_max - y_min
        desired_size = int(output_size * (1 - border_ratio))
        scale = desired_size / max(h, w)
        h2 = int(h * scale)
        w2 = int(w * scale)
        x2_min = (output_size - h2) // 2
        x2_max = x2_min + h2
        y2_min = (output_size - w2) // 2
        y2_max = y2_min + w2
        final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(carved_image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_depth[x2_min:x2_max, y2_min:y2_max] = cv2.resize(depth[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
        final_normal[x2_min:x2_max, y2_min:y2_max] = cv2.resize(normal[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    else:
        final_rgba = carved_image
        final_depth = depth
        final_normal = normal
    cv2.imwrite(out_rgba, cv2.cvtColor(final_rgba, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(out_depth, final_depth)
    cv2.imwrite(out_normal, final_normal)
main()