from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.BasicSR import *
exe = Executor('BasicSR', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/BasicSR/inference/inference_basicvsr.py'
import argparse
import cv2
import glob
import os
import shutil
import torch
from basicsr.archs.basicvsr_arch import BasicVSR
from basicsr.data.data_util import read_img_seq
from basicsr.utils.img_util import tensor2img

def inference(imgs, imgnames, model, save_path):
    with torch.no_grad():
        outputs = exe.run('forward', x=imgs)
    outputs = outputs.squeeze()
    outputs = list(outputs)
    for (output, imgname) in zip(outputs, imgnames):
        output = tensor2img(output)
        cv2.imwrite(os.path.join(FILE_RECORD_PATH, f'{imgname}_BasicVSR.png'), output)

def run_basic_vsr():
    model_path = 'experiments/pretrained_models/BasicVSR_REDS4.pth'
    input_path = 'datasets/REDS4/sharp_bicubic/000'
    save_path = 'results/BasicVSR'
    interval = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = exe.create_interface_objects(interface_class_name='BasicVSR', num_feat=64, num_block=30)
    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)
    os.makedirs(save_path, exist_ok=True)
    use_ffmpeg = False
    if not os.path.isdir(input_path):
        use_ffmpeg = True
        video_name = os.path.splitext(os.path.split(input_path)[-1])[0]
        input_path = os.path.join('./BasicVSR_tmp', video_name)
        os.makedirs(os.path.join('./BasicVSR_tmp', video_name), exist_ok=True)
        os.system(f'ffmpeg -i {input_path} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0 {input_path}/frame%08d.png')
    imgs_list = sorted(glob.glob(os.path.join(input_path, '*')))
    num_imgs = len(imgs_list)
    if len(imgs_list) <= interval:
        (imgs, imgnames) = read_img_seq(imgs_list, return_imgname=True)
        imgs = imgs.unsqueeze(0).to(device)
        inference(imgs, imgnames, model, save_path)
    else:
        for idx in range(0, num_imgs, interval):
            interval_size = min(interval, num_imgs - idx)
            (imgs, imgnames) = read_img_seq(imgs_list[idx:idx + interval_size], return_imgname=True)
            imgs = imgs.unsqueeze(0).to(device)
            inference(imgs, imgnames, model, save_path)
    if use_ffmpeg:
        shutil.rmtree(input_path)
run_basic_vsr()