import numpy as np
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite
from basicsr.utils import img2tensor
from basicsr.utils import tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
import warnings
from basicsr.utils.video_util import VideoReader
from basicsr.utils.video_util import VideoWriter
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.CodeFormer import ENV_DIR
from Inspection.adapters.custom_adapters.CodeFormer import *

exe = Executor('CodeFormer', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# 模拟参数
input_path = ENV_DIR+'/inputs/whole_imgs'
output_path = FILE_RECORD_PATH  # 使用 FILE_RECORD_PATH
fidelity_weight = 0.5
upscale = 2
has_aligned = False
only_center_face = False
draw_box = False
detection_model = 'retinaface_resnet50'
bg_upsampler = 'None'
face_upsample = False
bg_tile = 400
suffix = None
save_video_fps = None

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan():
    use_half = False
    if torch.cuda.is_available():  # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660']  # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path=ENV_DIR + "/weights/RealESRGAN_x2plus.pth",  # 使用 ENV_DIR
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )
    return upsampler

# ------------------------ main code ------------------------
device = get_device()

# ------------------------ input & output ------------------------
input_video = False
if input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):  # input single img path
    input_img_list = [input_path]
    result_root = FILE_RECORD_PATH + f'/test_results/test_img_{fidelity_weight}'  # 使用 FILE_RECORD_PATH
elif input_path.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')):  # input video path
    input_img_list = []
    vidreader = VideoReader(input_path)
    image = vidreader.get_frame()
    while image is not None:
        input_img_list.append(image)
        image = vidreader.get_frame()
    audio = vidreader.get_audio()
    fps = vidreader.get_fps() if save_video_fps is None else save_video_fps
    video_name = os.path.basename(input_path)[:-4]
    result_root = FILE_RECORD_PATH + f'/test_results/{video_name}_{fidelity_weight}'  # 使用 FILE_RECORD_PATH
    input_video = True
    vidreader.close()
else:  # input img folder
    if input_path.endswith('/'):  # solve when path ends with /
        input_path = input_path[:-1]
    # scan all the jpg and png images
    input_img_list = sorted(glob.glob(os.path.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    result_root = FILE_RECORD_PATH + f'/test_results/{os.path.basename(input_path)}_{fidelity_weight}'  # 使用 FILE_RECORD_PATH

if not output_path is None:  # set output path
    result_root = output_path

test_img_num = len(input_img_list)
if test_img_num == 0:
    raise FileNotFoundError('No input image/video is found...\n'
                            '\tNote that --input_path for video should end with .mp4|.mov|.avi')

# ------------------ set up background upsampler ------------------
if bg_upsampler == 'realesrgan':
    bg_upsampler = set_realesrgan()
else:
    bg_upsampler = None

# ------------------ set up face upsampler ------------------
if face_upsample:
    if bg_upsampler is not None:
        face_upsampler = bg_upsampler
    else:
        face_upsampler = set_realesrgan()
else:
    face_upsampler = None

# ------------------ set up CodeFormer restorer -------------------
net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                        connect_list=['32', '64', '128', '256']).to(device)

ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                model_dir=ENV_DIR + '/weights/CodeFormer', progress=True, file_name=None)  # 使用 ENV_DIR
checkpoint = torch.load(ckpt_path)['params_ema']
net.load_state_dict(checkpoint)
net.eval()

# ------------------ set up FaceRestoreHelper -------------------
exe.create_interface_objects(
    interface_class_name = 'FaceRestoreHelper',
    upscale_factor=upscale,
    face_size=512,
    crop_ratio=(1, 1),
    det_model=detection_model,
    save_ext='png',
    template_3points=False,
    pad_blur=False,
    use_parse=True,
    device=device
)

if not has_aligned:
    print(f'Face detection model: {detection_model}')
if bg_upsampler is not None:
    print(f'Background upsampling: True, Face upsampling: {face_upsample}')
else:
    print(f'Background upsampling: False, Face upsampling: {face_upsample}')

# -------------------- start to processing ---------------------
for i, img_path in enumerate(input_img_list):
    # clean all the intermediate results to process the next image
    exe.run("clean_all")
    
    if isinstance(img_path, str):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i + 1}/{test_img_num}] Processing: {img_name}')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:  # for video processing
        basename = str(i).zfill(6)
        img_name = f'{video_name}_{basename}' if input_video else basename
        print(f'[{i + 1}/{test_img_num}] Processing: {img_name}')
        img = img_path

    if has_aligned:
        # the input faces are already cropped and aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        exe.adapter.face_restore_helper.is_gray = is_gray(img, threshold=10)
        if exe.adapter.face_restore_helper.is_gray:
            print('Grayscale input: True')
        exe.adapter.face_restore_helper.cropped_faces = [img]
    else:
        exe.run("read_image", img=img)
        # get face landmarks for each face
        num_det_faces = exe.run("get_face_landmarks_5", only_keep_largest=only_center_face, only_center_face=only_center_face, resize=640, blur_ratio=0.01, eye_dist_threshold=5)
        print(f'\tdetect {num_det_faces} faces')
        # align and warp each face
        exe.run("align_warp_face")

    # face restoration for each cropped face
    for idx, cropped_face in enumerate(exe.adapter.face_restore_helper.cropped_faces):
        # prepare data
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        exe.run("add_restored_face", restored_face=restored_face, input_face=cropped_face)

    # paste_back
    if not has_aligned:
        # upsample the background
        if bg_upsampler is not None:
            # Now only support RealESRGAN for upsampling background
            bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
        else:
            bg_img = None
        exe.run("get_inverse_affine", save_inverse_affine_path=None)
        # paste each restored face to the input image
        if face_upsample and face_upsampler is not None:
            restored_img = exe.run("paste_faces_to_input_image", save_path=None, upsample_img=bg_img, draw_box=draw_box, face_upsampler=face_upsampler)
        else:
            restored_img = exe.run("paste_faces_to_input_image", save_path=None, upsample_img=bg_img, draw_box=draw_box)

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(exe.adapter.face_restore_helper.cropped_faces, exe.adapter.face_restore_helper.restored_faces)):
        # save cropped face
        if not has_aligned:
            save_crop_path = os.path.join(result_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
            imwrite(cropped_face, save_crop_path)
        # save restored face
        if has_aligned:
            save_face_name = f'{basename}.png'
        else:
            save_face_name = f'{basename}_{idx:02d}.png'
        if suffix is not None:
            save_face_name = f'{save_face_name[:-4]}_{suffix}.png'
        save_restore_path = os.path.join(result_root, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)

    # save restored img
    if not has_aligned and restored_img is not None:
        if suffix is not None:
            basename = f'{basename}_{suffix}'
        save_restore_path = os.path.join(result_root, 'final_results', f'{basename}.png')
        imwrite(restored_img, save_restore_path)

print(f'\nAll results are saved in {result_root}')
