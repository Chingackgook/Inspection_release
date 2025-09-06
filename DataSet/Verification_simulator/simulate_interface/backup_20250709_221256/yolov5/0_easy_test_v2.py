from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.yolov5 import ENV_DIR
from Inspection.adapters.custom_adapters.yolov5 import *
exe = Executor('yolov5', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode

# 可能需要手动修改的部分：
weights_path = Path(ENV_DIR + 'yolov5s.pt')  # 模型权重文件路径
source_path = Path(ENV_DIR + 'data/images' )   # 输入源路径
data_path = Path(ENV_DIR + 'data/coco128.yaml') # 数据集配置文件路径
conf_thres = 0.25                         # 置信度阈值
iou_thres = 0.45                          # IOU 阈值
max_det = 1000                            # 最大检测数量
save_csv = True                           # 是否保存 CSV
# end

@smart_inference_mode()
def run(weights=weights_path, source=source_path, data=data_path, imgsz=(640, 640), conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det, device='', view_img=False, save_txt=False, save_format=0, save_csv=save_csv, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=Path(FILE_RECORD_PATH + 'runs/detect'), name='exp', exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1):
    source = str(source)
    save_img = not nosave and (not source.endswith('.txt'))
    is_file = Path(source).suffix[1:] in IMG_FORMATS + VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and (not is_file))
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    device = select_device(device)
    model = exe.create_interface_objects(interface_class_name='DetectMultiBackend', weights=weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = (model.stride, model.names, model.pt)
    imgsz = check_img_size(imgsz, s=stride)
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = ([None] * bs, [None] * bs)
    exe.run('warmup', imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = (0, [], (Profile(device=device), Profile(device=device), Profile(device=device)))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = exe.run('forward', im=image, augment=augment, visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, exe.run('forward', im=image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = exe.run('forward', im=im, augment=augment, visualize=visualize)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        csv_path = save_dir / 'predictions.csv'

        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
        for i, det in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = (path[i], im0s[i].copy(), dataset.count)
                s += f'{i}: '
            else:
                p, im0, frame = (path, im0s.copy(), getattr(dataset, 'frame', 0))
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '{:g}x{:g} '.format(*im.shape[2:])
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f'{n} {names[int(c)]}{'s' * (n > 1)}, '
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'
                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)
                    if save_txt:
                        if save_format == 0:
                            coords = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        else:
                            coords = (torch.tensor(xyxy).view(1, 4) / gn).view(-1).tolist()
                        line = (cls, *coords, conf) if save_conf else (cls, *coords)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:
                        c = int(cls)
                        label = None if hide_labels else names[c] if hide_conf else f'{names[c]} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = (30, im0.shape[1], im0.shape[0])
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        LOGGER.info(f'{s}{('' if len(det) else '(no detections), ')}{dt[1].dt * 1000.0:.1f}ms')
    t = tuple((x.t / seen * 1000.0 for x in dt))
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f'\n{len(list(save_dir.glob("labels/*.txt")))} labels saved to {save_dir / "labels"}' if save_txt else ''
        LOGGER.info(f'Results saved to {colorstr("bold", save_dir)}{s}')
    if update:
        strip_optimizer(weights[0])

def main():
    run()

# 直接运行主逻辑
main()
