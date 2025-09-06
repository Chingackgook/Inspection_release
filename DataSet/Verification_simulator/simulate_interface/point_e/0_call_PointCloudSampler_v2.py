from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.point_e import *
import torch
from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud
exe = Executor('point_e', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/point-e/point_e/examples/text2pointcloud.ipynb'

def run_point_cloud_sampling():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    print('Creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    print('Downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))
    print('Downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    sampler = exe.create_interface_objects(interface_class_name='PointCloudSampler', device=device, models=[base_model, upsampler_model], diffusions=[base_diffusion, upsampler_diffusion], num_points=[1024, 4096 - 1024], aux_channels=['R', 'G', 'B'], guidance_scale=[3.0, 0.0], model_kwargs_key_filter=('texts', ''))
    prompt = 'a red motorcycle'
    samples = None
    for x in tqdm(exe.run('sample_batch_progressive', batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
    pc = exe.run('output_to_point_clouds', output=samples)[0]
    fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))
run_point_cloud_sampling()