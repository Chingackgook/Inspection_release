from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.pulse import *
exe = Executor('pulse', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/pulse/run.py'
from PULSE import PULSE
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from pathlib import Path
from PIL import Image
import torchvision
from math import log10
from math import ceil
import argparse
import torch
from PULSE import PULSE
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision
from math import log10, ceil
import torch

class Images(Dataset):

    def __init__(self, root_dir, duplicates):
        self.root_path = Path(root_dir)
        self.image_list = list(self.root_path.glob('*.png'))
        self.duplicates = duplicates

    def __len__(self):
        return self.duplicates * len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx // self.duplicates]
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.duplicates == 1:
            return (image, img_path.stem)
        else:
            return (image, img_path.stem + f'_{idx % self.duplicates + 1}')
input_dir = 'input'
output_dir = 'runs'
cache_dir = 'cache'
duplicates = 1
batch_size = 1
seed = None
loss_str = '100*L2+0.05*GEOCROSS'
eps = 0.002
noise_type = 'trainable'
num_trainable_noise_layers = 5
tile_latent = False
bad_noise_layers = '17'
opt_name = 'adam'
learning_rate = 0.4
steps = 100
lr_schedule = 'linear1cycledrop'
save_intermediate = False
kwargs = {'input_dir': input_dir, 'output_dir': output_dir, 'cache_dir': cache_dir, 'duplicates': duplicates, 'batch_size': batch_size, 'seed': seed, 'loss_str': loss_str, 'eps': eps, 'noise_type': noise_type, 'num_trainable_noise_layers': num_trainable_noise_layers, 'tile_latent': tile_latent, 'bad_noise_layers': bad_noise_layers, 'opt_name': opt_name, 'learning_rate': learning_rate, 'steps': steps, 'lr_schedule': lr_schedule, 'save_intermediate': save_intermediate}
dataset = Images(kwargs['input_dir'], duplicates=kwargs['duplicates'])
out_path = Path(kwargs['output_dir'])
out_path.mkdir(parents=True, exist_ok=True)
dataloader = DataLoader(dataset, batch_size=kwargs['batch_size'])
model = exe.create_interface_objects(interface_class_name='PULSE', cache_dir=kwargs['cache_dir'], verbose=True)
toPIL = torchvision.transforms.ToPILImage()
for ref_im, ref_im_name in dataloader:
    if torch.cuda.is_available():
        ref_im = ref_im.cuda()
    if kwargs['save_intermediate']:
        padding = ceil(log10(100))
        for i in range(kwargs['batch_size']):
            int_path_HR = Path(FILE_RECORD_PATH) / ref_im_name[i] / 'HR'
            int_path_LR = Path(FILE_RECORD_PATH) / ref_im_name[i] / 'LR'
            int_path_HR.mkdir(parents=True, exist_ok=True)
            int_path_LR.mkdir(parents=True, exist_ok=True)
        for j, (HR, LR) in enumerate(exe.run('forward', ref_im=ref_im, seed=kwargs['seed'], loss_str=kwargs['loss_str'], eps=kwargs['eps'], noise_type=kwargs['noise_type'], num_trainable_noise_layers=kwargs['num_trainable_noise_layers'], tile_latent=kwargs['tile_latent'], bad_noise_layers=kwargs['bad_noise_layers'], opt_name=kwargs['opt_name'], learning_rate=kwargs['learning_rate'], steps=kwargs['steps'], lr_schedule=kwargs['lr_schedule'], save_intermediate=kwargs['save_intermediate'])):
            for i in range(kwargs['batch_size']):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(int_path_HR / f'{ref_im_name[i]}_{j:0{padding}}.png')
                toPIL(LR[i].cpu().detach().clamp(0, 1)).save(int_path_LR / f'{ref_im_name[i]}_{j:0{padding}}.png')
    else:
        for j, (HR, LR) in enumerate(exe.run('forward', ref_im=ref_im, seed=kwargs['seed'], loss_str=kwargs['loss_str'], eps=kwargs['eps'], noise_type=kwargs['noise_type'], num_trainable_noise_layers=kwargs['num_trainable_noise_layers'], tile_latent=kwargs['tile_latent'], bad_noise_layers=kwargs['bad_noise_layers'], opt_name=kwargs['opt_name'], learning_rate=kwargs['learning_rate'], steps=kwargs['steps'], lr_schedule=kwargs['lr_schedule'], save_intermediate=kwargs['save_intermediate'])):
            for i in range(kwargs['batch_size']):
                toPIL(HR[i].cpu().detach().clamp(0, 1)).save(Path(FILE_RECORD_PATH) / f'{ref_im_name[i]}.png')