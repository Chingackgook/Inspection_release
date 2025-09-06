from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.RobustVideoMatting import *
exe = Executor('RobustVideoMatting', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0
    ] = '/mnt/autor_name/haoTingDeWenJianJia/RobustVideoMatting/inference.py'
import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm
from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
from model import MattingNetwork


def convert_video(model, input_source: str='input.mp4', input_resize:
    Optional[Tuple[int, int]]=None, downsample_ratio: Optional[float]=None,
    output_type: str='video', output_composition: Optional[str]=os.path.
    join(FILE_RECORD_PATH, 'composition.mp4'), output_alpha: Optional[str]=
    os.path.join(FILE_RECORD_PATH, 'alpha.mp4'), output_foreground:
    Optional[str]=os.path.join(FILE_RECORD_PATH, 'foreground.mp4'),
    output_video_mbps: Optional[float]=4, seq_chunk: int=1, num_workers:
    int=0, progress: bool=True, device: Optional[str]='cuda', dtype:
    Optional[torch.dtype]=None):
    assert downsample_ratio is None or downsample_ratio > 0 and downsample_ratio <= 1, 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground]
        ), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'
        ], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    if input_resize is not None:
        transform = transforms.Compose([transforms.Resize(input_resize[::-1
            ]), transforms.ToTensor()])
    else:
        transform = transforms.ToTensor()
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True,
        num_workers=num_workers)
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader
            ) else 30
        output_video_mbps = (1 if output_video_mbps is None else
            output_video_mbps)
        if output_composition is not None:
            writer_com = VideoWriter(path=output_composition, frame_rate=
                frame_rate, bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(path=output_alpha, frame_rate=
                frame_rate, bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(path=output_foreground, frame_rate=
                frame_rate, bit_rate=int(output_video_mbps * 1000000))
    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_alpha, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    if output_composition is not None and output_type == 'video':
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255
            ).view(1, 1, 3, 1, 1)
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress,
                dynamic_ncols=True)
            rec = [None] * 4
            for src in reader:
                if downsample_ratio is None:
                    downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                src = src.to(device, dtype, non_blocking=True).unsqueeze(0)
                fgr, pha, *rec = exe.run('forward', src=src, r1=rec[0], r2=
                    rec[1], r3=rec[2], r4=rec[3], downsample_ratio=
                    downsample_ratio)
                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                if output_alpha is not None:
                    writer_pha.write(pha[0])
                if output_composition is not None:
                    if output_type == 'video':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])
                bar.update(src.size(1))
    finally:
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:

    # manual add checkpoint path
    def __init__(self, variant: str='mobilenetv3', checkpoint: str='/mnt/autor_name/haoTingDeWenJianJia/RobustVideoMatting/rvm_mobilenetv3.pth',
        device: str='cuda'):
        self.model = exe.create_interface_objects(interface_class_name=
            'MattingNetwork', variant='mobilenetv3', refiner=
            'deep_guided_filter', pretrained_backbone=False).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

    def convert(self, *args, **kwargs):
        convert_video(self.model, *args, device=self.device, dtype=torch.
            float32, **kwargs)


def run_conversion():
    converter = Converter()
    # manual add input source
    converter.convert(input_source='/mnt/autor_name/Inspection/Resources/videos/test_video.mp4', output_type='video',
        output_composition=os.path.join(FILE_RECORD_PATH, 'composition.mp4'
        ), output_alpha=os.path.join(FILE_RECORD_PATH, 'alpha.mp4'),
        output_foreground=os.path.join(FILE_RECORD_PATH, 'foreground.mp4'),
        output_video_mbps=4, seq_chunk=1, num_workers=0, progress=True)


run_conversion()
