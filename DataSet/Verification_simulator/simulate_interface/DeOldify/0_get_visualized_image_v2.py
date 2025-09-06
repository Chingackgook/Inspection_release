from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.DeOldify import *
exe = Executor('DeOldify','simulation')
FILE_RECORD_PATH = exe.now_record_path
# Import the existing package


import os
import statistics
from fastai import *
from deoldify.visualize import *
from deoldify.visualize import ModelImageVisualizer
import cv2
from fid.fid_score import *
from fid.inception import *
import imageio
import warnings
import PIL
from fastai.vision import *
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
plt.style.use('dark_background')
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')
warnings.filterwarnings('ignore', category=UserWarning, message='.*?retrieve source code for container of type.*?')
# add
path = Path(ENV_DIR + '/data/ColorBenchmark')
# end add
# origin code:
#path = Path('data/ColorBenchmark')
path_hr = path / 'source'
path_lr = path / 'bandw'
path_results = Path(FILE_RECORD_PATH) / 'result_images/ColorBenchmarkFID/artistic'  # Updated to use FILE_RECORD_PATH
path_rendered = path_results / 'rendered'
num_images = 50000
render_factor = 35
fid_batch_size = 4
eval_size = 299

def get_image_colorizer(root_folder: Path=Path('./'), render_factor: int=35, artistic: bool=True) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)

def inception_model(dims: int):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    return model

def create_before_images(fn, i):
    dest = path_lr / fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn).convert('LA').convert('RGB')
    img.save(dest)

def render_images(colorizer, source_dir: Path, filtered_dir: Path, target_dir: Path, render_factor: int, num_images: int) -> [(Path, Path, Path)]:
    results = []
    bandw_list = ImageList.from_folder(path_lr)
    bandw_list = bandw_list[:num_images]
    if len(bandw_list.items) == 0:
        return results
    img_iterator = progress_bar(bandw_list.items)
    for bandw_path in img_iterator:
        target_path = target_dir / bandw_path.relative_to(source_dir)
        try:
            result_image = exe.run('get_transformed_image', path=bandw_path, render_factor=render_factor)
            result_path = path_results / bandw_path.parent.name / bandw_path.name  # Updated to use FILE_RECORD_PATH
            if not result_path.parent.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(result_path)
            results.append((result_path, bandw_path, target_path))
        except Exception as err:
            print('Failed to render image.  Skipping.  Details: {0}'.format(err))
    return results

def calculate_fid_score(render_results, bs: int, eval_size: int):
    dims = 2048
    cuda = True
    model = inception_model(dims=dims)
    rendered_paths = []
    target_paths = []
    for render_result in render_results:
        (rendered_path, _, target_path) = render_result
        rendered_paths.append(str(rendered_path))
        target_paths.append(str(target_path))
    (rendered_m, rendered_s) = calculate_activation_statistics(files=rendered_paths, model=model, batch_size=bs, dims=dims, cuda=cuda)
    (target_m, target_s) = calculate_activation_statistics(files=target_paths, model=model, batch_size=bs, dims=dims, cuda=cuda)
    fid_score = calculate_frechet_distance(rendered_m, rendered_s, target_m, target_s)
    del model
    return fid_score

# Check if the low-resolution path exists and create images if it doesn't
if not path_lr.exists():
    il = ImageList.from_folder(path_hr)
    parallel(create_before_images, il.items)

path_results.parent.mkdir(parents=True, exist_ok=True)
# add
colorizer = get_image_colorizer(artistic=True)
exe.adapter.class1_obj = colorizer
# add end

#origin code:
#colorizer = exe.create_interface_objects(interface_class_name='ModelImageVisualizer', filter=my_filter, results_dir='output_directory')
render_results = render_images(colorizer=colorizer, source_dir=path_lr, target_dir=path_hr, filtered_dir=path_results, render_factor=render_factor, num_images=num_images)
fid_score = calculate_fid_score(render_results, bs=fid_batch_size, eval_size=eval_size)
print('FID Score: ' + str(fid_score))
