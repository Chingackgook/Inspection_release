from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.DeOldify import ENV_DIR
from Inspection.adapters.custom_adapters.DeOldify import *
exe = Executor('DeOldify','simulation')
FILE_RECORD_PATH = exe.now_record_path

# 导入原有的包
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
import matplotlib.pyplot as plt

# 可能需要手动修改的部分：
num_images = 50000  # 你可以根据需要调整要处理的图像数量
render_factor = 35  # 渲染因子，根据需求调整
fid_batch_size = 4  # FID计算的批量大小，可能需要根据显存进行调整
# end

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
plt.style.use('dark_background')
torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore', category=UserWarning, module='torch.nn.functional')
warnings.filterwarnings('ignore', category=UserWarning, message='.*?retrieve source code for container of type.*?')

# 使用全局变量ENV_DIR替换输入路径
path = Path(ENV_DIR) / 'ColorBenchmark'
path_hr = path / 'source'  # 输入路径：高分辨率图像的路径
path_lr = path / 'bandw'   # 输入路径：低分辨率图像的路径

# 使用全局变量FILE_RECORD_PATH替换输出路径
path_results = Path(FILE_RECORD_PATH) / 'result_images/ColorBenchmarkFID/artistic'  # 输出路径：结果图像的存储路径
path_rendered = path_results / 'rendered'

def get_image_colorizer(root_folder: Path=Path('./'), render_factor: int=35, artistic: bool=True) -> ModelImageVisualizer:
    if artistic:
        return get_artistic_image_colorizer(root_folder=root_folder, render_factor=render_factor)
    else:
        return get_stable_image_colorizer(root_folder=root_folder, render_factor=render_factor)

def inception_model(dims: int):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.cuda()
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
            result_path = path_results / bandw_path.parent.name / bandw_path.name  # 使用输出路径
            if not result_path.parent.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
            result_image.save(result_path)
            results.append((result_path, bandw_path, target_path))
        except Exception as err:
            print('Failed to render image. Skipping. Details: {0}'.format(err))
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

# 主逻辑
if not path_lr.exists():
    il = ImageList.from_folder(path_hr)
    parallel(create_before_images, il.items)
path_results.parent.mkdir(parents=True, exist_ok=True)
exe.create_interface_objects(interface_class_name='ModelImageVisualizer', filter=None, results_dir=None)
colorizer = get_image_colorizer(artistic=True)
render_results = render_images(colorizer=colorizer, source_dir=path_lr, target_dir=path_hr, filtered_dir=path_results, render_factor=render_factor, num_images=num_images)
fid_score = calculate_fid_score(render_results, bs=fid_batch_size, eval_size=eval_size)
print('FID Score: ' + str(fid_score))
