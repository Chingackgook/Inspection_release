from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.yolov7 import *
exe = Executor('yolov7', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/yolov7/detect.py'
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, increment_path, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, TracedModel, time_synchronized

def detect(save_img=False):

    class Opt:

        def __init__(self):
            self.source = 'inference/images'
            self.weights = 'yolov7.pt'
            self.view_img = False
            self.save_txt = False
            self.img_size = 640
            self.no_trace = False
            self.nosave = False
            self.device = ''
            self.project = 'runs/detect'
            self.name = 'exp'
            self.exist_ok = True
            self.conf_thres = 0.25
            self.iou_thres = 0.45
            self.classes = None
            self.agnostic_nms = False
            self.augment = False
    opt = Opt()
    save_img = not opt.nosave and (not opt.source.endswith('.txt'))
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = Path(increment_path(Path(FILE_RECORD_PATH) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(opt.weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(opt.img_size, s=stride)
    if not opt.no_trace:
        model = exe.create_interface_objects(interface_class_name='TracedModel', model=model, device=device, img_size=opt.img_size)
    if half:
        model.half()
    vid_path, vid_writer = (None, None)
    if webcam:
        view_img = False
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
        t1 = time_synchronized()
        with torch.no_grad():
            pred = exe.run('forward', x=img, augment=opt.augment)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = (path[i], '%g: ' % i, im0s[i].copy(), dataset.count)
            else:
                p, s, im0, frame = (path, '', im0s, getattr(dataset, 'frame', 0))
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                for *xyxy, conf, cls in reversed(det):
                    if opt.save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
            print(f'{s}Done. ({1000.0 * (t2 - t1):.1f}ms) Inference, ({1000.0 * (t3 - t2):.1f}ms) NMS')
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f' The image with the result is saved in: {save_path}')
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = (30, im0.shape[1], im0.shape[0])
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
    if opt.save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}' if opt.save_txt else '"
        print(f'Results saved to {save_dir}{s}')
    print(f'Done. ({time.time() - t0:.3f}s)')
with torch.no_grad():
    detect()