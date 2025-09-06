from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.StreamDiffusion import *
exe = Executor('StreamDiffusion', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/StreamDiffusion/examples/img2img/single.py'
import os
import sys
from typing import Literal
from typing import Dict
from typing import Optional
import fire
from utils.wrapper import StreamDiffusionWrapper
import os
import sys
from typing import Literal, Dict, Optional
from utils.wrapper import StreamDiffusionWrapper
CURRENT_DIR = os.path.dirname(os.path.abspath('/mnt/autor_name/haoTingDeWenJianJia/StreamDiffusion/examples/img2img/single.py'))

def main(input: str=os.path.join(CURRENT_DIR, '..', '..', 'images', 'inputs', 'input.png'), output: str=os.path.join(FILE_RECORD_PATH, 'output.png'), model_id_or_path: str='KBlueLeaf/kohaku-v2.1', lora_dict: Optional[Dict[str, float]]=None, prompt: str='1girl with brown dog hair, thick glasses, smiling', negative_prompt: str='low quality, bad quality, blurry, low resolution', width: int=512, height: int=512, acceleration: Literal['none', 'xformers', 'tensorrt']='xformers', use_denoising_batch: bool=True, guidance_scale: float=1.2, cfg_type: Literal['none', 'full', 'self', 'initialize']='self', seed: int=2, delta: float=0.5):
    if guidance_scale <= 1.0:
        cfg_type = 'none'
    stream = exe.create_interface_objects(interface_class_name='StreamDiffusionWrapper', model_id_or_path=model_id_or_path, lora_dict=lora_dict, t_index_list=[22, 32, 45], frame_buffer_size=1, width=width, height=height, warmup=10, acceleration=acceleration, mode='img2img', use_denoising_batch=use_denoising_batch, cfg_type=cfg_type, seed=seed)
    var = exe.run('prepare', prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=50, guidance_scale=guidance_scale, delta=delta)
    image_tensor = stream.preprocess_image(input)
    for _ in range(stream.batch_size - 1):
        var = exe.run('__call__', image=image_tensor)
    output_image = exe.run('__call__', image=image_tensor)
    output_image.save(output)
main()