from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.latent_diffusion import *
exe = Executor('latent_diffusion', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/latent-diffusion/scripts/txt2img.py'
import argparse
import os
import sys
import glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config

def load_model_from_config(config, ckpt, verbose=False):
    print(f'Loading model from {ckpt}')
    pl_sd = torch.load(ckpt, map_location='cpu')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('unexpected keys:')
        print(u)
    model.cuda()
    model.eval()
    return model

def run_image_generation():
    prompt = 'a painting of a virus monster playing guitar'
    outdir = FILE_RECORD_PATH
    ddim_steps = 200
    plms = False
    ddim_eta = 0.0
    n_iter = 1
    H = 256
    W = 256
    n_samples = 4
    scale = 5.0
    config = OmegaConf.load('configs/latent-diffusion/txt2img-1p4B-eval.yaml')
    model = load_model_from_config(config, 'models/ldm/text2img-large/model.ckpt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if plms:
        sampler = PLMSSampler(model=model)
    else:
        sampler = exe.create_interface_objects(interface_class_name='DDIMSampler', model=model)
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    all_samples = list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [''])
            for n in trange(n_iter, desc='Sampling'):
                c = model.get_learned_conditioning(n_samples * [prompt])
                shape = [4, H // 8, W // 8]
                samples_ddim, _ = exe.run('sample', S=ddim_steps, batch_size=n_samples, shape=shape, conditioning=c, verbose=False, unconditional_guidance_scale=scale, unconditional_conditioning=uc, eta=ddim_eta)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                for x_sample in x_samples_ddim:
                    x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:04}.png'))
                    base_count += 1
                all_samples.append(x_samples_ddim)
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples)
    grid = 255.0 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f"{prompt.replace(' ', '-')}.png"))
    print(f'Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.')
run_image_generation()