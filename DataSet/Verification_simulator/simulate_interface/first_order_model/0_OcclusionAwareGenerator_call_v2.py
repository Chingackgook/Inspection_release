from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.first_order_model import *
exe = Executor('first_order_model', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/first-order-model/demo.py'
import sys
import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import ffmpeg
from os.path import splitext
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
import face_alignment
from scipy.spatial import ConvexHull
if sys.version_info[0] < 3:
    raise Exception('You must use Python 3 or higher. Recommended version is Python 3.7')

def load_checkpoints(config_path, checkpoint_path, cpu=False):
    with open(config_path) as f:
        config = yaml.full_load(f)
    generator = exe.create_interface_objects(interface_class_name='OcclusionAwareGenerator', **config['model_params']['generator_params'], **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'], **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
    generator.eval()
    kp_detector.eval()
    return (generator, kp_detector)

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving, kp_driving_initial=kp_driving_initial, use_relative_movement=relative, use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = exe.run('forward', source_image=source, kp_source=kp_source, kp_driving=kp_norm)
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment
    from scipy.spatial import ConvexHull
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

# add
# manually set the paths for the config , checkpoint
def run_animation(config_path='config/bair-256.yaml', checkpoint_path='checkpoints2/fashion.pth.tar', source_image_path='sup-mat/source.png', driving_video_path='driving.mp4', result_video_path=FILE_RECORD_PATH + '/result.mp4', relative=False, adapt_scale=False, find_best_frame=False, best_frame=None, cpu=False, audio=False):
    source_image = imageio.imread(source_image_path)
    reader = imageio.get_reader(driving_video_path)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(config_path=config_path, checkpoint_path=checkpoint_path, cpu=cpu)
    if find_best_frame or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print('Best frame: ' + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:i + 1][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=relative, adapt_movement_scale=adapt_scale, cpu=cpu)
    imageio.mimsave(result_video_path, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    if audio:
        try:
            with NamedTemporaryFile(suffix=splitext(result_video_path)[1]) as output:
                ffmpeg.output(ffmpeg.input(result_video_path).video, ffmpeg.input(driving_video_path).audio, output.name, c='copy').run()
                with open(result_video_path, 'wb') as result:
                    copyfileobj(output, result)
        except ffmpeg.Error:
            print('Failed to copy audio: the driving video may have no audio track or the audio format is invalid.')
run_animation()