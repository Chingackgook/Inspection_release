from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.cogvideo import *
exe = Executor('cogvideo', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/CogVideo/inference/cli_demo.py'
import argparse
import logging
from typing import Literal
from typing import Optional
import torch
from diffusers import CogVideoXDPMScheduler
from diffusers import CogVideoXImageToVideoPipeline
from diffusers import CogVideoXPipeline
from diffusers import CogVideoXVideoToVideoPipeline
from diffusers.utils import export_to_video
from diffusers.utils import load_image
from diffusers.utils import load_video
import logging
from typing import Literal, Optional
import torch
from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline, CogVideoXPipeline, CogVideoXVideoToVideoPipeline
from diffusers.utils import export_to_video, load_image, load_video
logging.basicConfig(level=logging.INFO)
RESOLUTION_MAP = {'cogvideox1.5-5b-i2v': (768, 1360), 'cogvideox1.5-5b': (768, 1360), 'cogvideox-5b-i2v': (480, 720), 'cogvideox-5b': (480, 720), 'cogvideox-2b': (480, 720)}

def generate_video(prompt: str, model_path: str, lora_path: str=None, lora_rank: int=128, num_frames: int=81, width: Optional[int]=None, height: Optional[int]=None, output_path: str=FILE_RECORD_PATH + '/output.mp4', image_or_video_path: str='', num_inference_steps: int=50, guidance_scale: float=6.0, num_videos_per_prompt: int=1, dtype: torch.dtype=torch.bfloat16, generate_type: str='t2v', seed: int=42, fps: int=16):
    """
    Generates a video based on the given prompt and saves it to the specified path.
    """
    pass

def run_video_generation():
    prompt = 'A girl riding a bike.'
    model_path = 'THUDM/CogVideoX-2b'
    lora_path = None
    lora_rank = 128
    output_path = FILE_RECORD_PATH + '/output.mp4'
    num_frames = 81
    width = None
    height = None
    guidance_scale = 6.0
    num_inference_steps = 50
    image_or_video_path = RESOURCES_PATH + 'images/test_image.jpg'
    num_videos_per_prompt = 1
    dtype = torch.bfloat16
    generate_type = 't2v'
    seed = 42
    fps = 16
    exe.run('generate_video', prompt=prompt, model_path=model_path, lora_path=lora_path, lora_rank=lora_rank, output_path=output_path, num_frames=num_frames, width=width, height=height, image_or_video_path=image_or_video_path, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, num_videos_per_prompt=num_videos_per_prompt, dtype=dtype, generate_type=generate_type, seed=seed, fps=fps)
run_video_generation()