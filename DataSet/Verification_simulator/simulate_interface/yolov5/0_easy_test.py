import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.yolov5 import ENV_DIR
from Inspection.adapters.custom_adapters.yolov5 import *
exe = Executor('yolov5', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import os
import platform
import sys
from pathlib import Path
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run():
    weights = os.path.join(ENV_DIR, "yolov5s.pt")  # 模型路径
    source = os.path.join(ENV_DIR, "data/images")  # 输入源
    data = os.path.join(ENV_DIR, "data/coco128.yaml")  # 数据集配置文件路径
    imgsz = (640, 640)  # 输入图像尺寸
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # NMS IOU 阈值
    max_det = 1000  # 每张图像的最大检测数
    view_img = False  # 是否显示结果
    save_txt = False  # 是否保存结果到 *.txt
    save_csv = False  # 是否保存结果到 CSV 格式
    save_img = True  # 是否保存推理图像
    classes = None  # 过滤类别
    augment = False  # 是否进行增强推理
    visualize = False  # 是否可视化特征
    device = torch.device("cpu")  # 设备选择
    project = os.path.join(FILE_RECORD_PATH, "runs/detect")  # 保存结果的项目路径
    name = "exp"  # 保存结果的名称
    exist_ok = False  # 是否允许存在相同名称的项目

    # Load model
    exe.create_interface_objects(weights=weights)  # 加载模型
    model = exe.adapter.model  # 从 exe 对象中获取模型
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    exe.run("warmup", imgsz=(1, 3, 640, 640))  # 预热模型
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = exe.run("from_numpy", x=im)  # 将 NumPy 数组转换为 PyTorch 张量
            im = im.half() if model.fp16 else im.float()  # uint8 转为 fp16/32
            im /= 255  # 归一化
            if len(im.shape) == 3:
                im = im[None]  # 扩展为批次维度

        # Inference
        with dt[1]:
            visualize = increment_path(Path(project) / Path(path).stem, mkdir=True) if visualize else False
            pred = exe.run("forward", im=im, augment=augment, visualize=visualize)  # 执行推理

        #
        # 处理预测结果
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            # 处理检测结果
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                annotator = Annotator(im0, line_width=3, example=str(names))
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c]
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

    # Print results
    LOGGER.info(f"Results saved to {colorstr('bold', project)}")

run()
