from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.magic_animate import *
exe = Executor('magic_animate', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/magic-animate/demo/gradio_animate.py'
import argparse
import imageio
import numpy as np
import gradio as gr
from PIL import Image
from demo.animate import MagicAnimate
import imageio
import numpy as np
from PIL import Image
animator = exe.create_interface_objects(interface_class_name='MagicAnimate', config='configs/prompts/animation.yaml')

def animate(reference_image, motion_sequence_state, seed, steps, guidance_scale):
    return exe.run('__call__', source_image=reference_image, motion_sequence=motion_sequence_state, random_seed=seed, step=steps, guidance_scale=guidance_scale)
reference_image_path = 'inputs/applications/source_image/monalisa.png'
motion_sequence_path = 'inputs/applications/driving/densepose/running.mp4'
random_seed = 1
sampling_steps = 25
guidance_scale = 7.5

def read_image(image_path, size=512):
    image = Image.open(image_path)
    return np.array(image.resize((size, size)))

def read_video(video_path):
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']
    return video_path
reference_image = read_image(reference_image_path)
motion_sequence = read_video(motion_sequence_path)
animation_path = animate(reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale)
animation_path = FILE_RECORD_PATH + '/' + animation_path.split('/')[-1]
print(f'Generated animation saved at: {animation_path}')