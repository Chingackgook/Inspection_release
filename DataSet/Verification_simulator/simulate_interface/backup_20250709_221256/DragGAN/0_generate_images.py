import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.DragGAN import ENV_DIR
from Inspection.adapters.custom_adapters.DragGAN import *
exe = Executor('DragGAN', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
import os
import re
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import click
import dnnlib
import numpy
import PIL.Image
import torch
import legacy

# 模拟参数定义
network_pkl = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl"
seeds = [0, 1, 2]  # 模拟随机种子列表
truncation_psi = 1.0  # 模拟截断参数
noise_mode = 'const'  # 模拟噪声模式
outdir = FILE_RECORD_PATH + '/output_images'  # 使用FILE_RECORD_PATH替换输出目录
translate = (0.0, 0.0)  # 模拟平移坐标
rotate = 0.0  # 模拟旋转角度
class_idx = None  # 模拟类标签

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float, float],
    rotate: float,
    class_idx: Optional[int]
):
    """Generate images using the pretrained network pickle."""
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    dtype = torch.float32 if device.type == 'mps' else torch.float64
    with legacy.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device, dtype=dtype)

    os.makedirs(outdir, exist_ok=True)
    
    G = G.to(torch.float32)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise ValueError('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device, dtype=dtype)

        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

exe.run("generate_images", network_pkl=network_pkl, seeds=seeds, truncation_psi=truncation_psi, noise_mode=noise_mode, outdir=outdir, translate=translate, rotate=rotate, class_idx=class_idx)
