from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.consistency_models import *
exe = Executor('consistency_models', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/consistency_models/scripts/image_sample.py'
import argparse
import os
import numpy as np
import torch as th
import torch.distributed as dist
from cm import dist_util
from cm import logger
from cm.script_util import NUM_CLASSES
from cm.script_util import model_and_diffusion_defaults
from cm.script_util import create_model_and_diffusion
from cm.script_util import add_dict_to_argparser
from cm.script_util import args_to_dict
from cm.random_util import get_generator
from cm.karras_diffusion import karras_sample
'\nGenerate a large batch of image samples from a model and save them as a large\nnumpy array. This can be used to produce samples for FID evaluation.\n'
import os
import numpy as np
import torch as th
import torch.distributed as dist
from cm import dist_util, logger
from cm.script_util import NUM_CLASSES, model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict
from cm.random_util import get_generator

def run_sampling():

    class Args:
        training_mode = 'edm'  # add
        generator = 'determ-indiv'  # add
        batch_size = 8  # add
        sigma_max = 80  # add
        sigma_min = 0.002  # add
        s_churn = 0  # add
        steps = 40  # add
        sampler = 'heun'  # add
        model_path = '/mnt/autor_name/haoTingDeWenJianJia/consistency_models/edm_bedroom256_ema.pt'  # add
        attention_resolutions = '32,16,8'  # add
        class_cond = False  # add
        dropout = 0.1  # add
        image_size = 256  # add
        num_channels = 256  # add
        num_head_channels = 64  # add
        num_res_blocks = 2  # add
        num_samples = 16  # add
        resblock_updown = True  # add
        use_fp16 = True  # add
        use_scale_shift_norm = False  # add
        weight_schedule = 'karras'  # add
        s_tmin = 0.0  # add
        s_tmax = float('inf')  # add
        s_noise = 1.0  # add

        num_heads = 4  # add
        num_heads_upsample = -1  # add
        channel_mult = ""  # add
        use_checkpoint = False  # add
        use_new_attention_order = False  # add
        learn_sigma = False  # add

        clip_denoised = True
        ts = ''
        seed = 42

    args = Args()
    dist_util.setup_dist()
    logger.configure()
    if 'consistency' in args.training_mode:
        distillation = True
    else:
        distillation = False
    logger.log('creating model and diffusion...')
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()), distillation=distillation)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location='cpu'))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log('sampling...')
    if args.sampler == 'multistep':
        ts = tuple((int(x) for x in args.ts.split(','))) if args.ts else None
    else:
        ts = None
    all_images = []
    all_labels = []
    generator = get_generator(args.generator, args.num_samples, args.seed)
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())
            model_kwargs['y'] = classes
        sample = exe.run('karras_sample', diffusion=diffusion, model=model, shape=(args.batch_size, 3, args.image_size, args.image_size), steps=args.steps, clip_denoised=args.clip_denoised, model_kwargs=model_kwargs, device=dist_util.dev(), sigma_min=args.sigma_min, sigma_max=args.sigma_max, s_churn=args.s_churn, s_tmin=args.s_tmin, s_tmax=args.s_tmax, s_noise=args.s_noise, generator=generator, ts=ts)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f'created {len(all_images) * args.batch_size} samples')
    arr = np.concatenate(all_images, axis=0)
    arr = arr[:args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[:args.num_samples]
    if dist.get_rank() == 0:
        shape_str = 'x'.join([str(x) for x in arr.shape])
        out_path = os.path.join(FILE_RECORD_PATH, f'samples_{shape_str}.npz')
        logger.log(f'saving to {out_path}')
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
    dist.barrier()
    logger.log('sampling complete')
run_sampling()