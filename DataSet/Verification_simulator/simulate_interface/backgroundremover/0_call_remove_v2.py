from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.backgroundremover import *
exe = Executor('backgroundremover', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/backgroundremover/backgroundremover/cmd/cli.py'
import argparse
import os
from distutils.util import strtobool
from backgroundremover import utilities
from backgroundremover.bg import remove

def execute_background_removal():
    model_choices = ['u2net', 'u2net_human_seg', 'u2netp']
    args = {'model': 'u2net', 'alpha_matting': False, 'alpha_matting_foreground_threshold': 240, 'alpha_matting_background_threshold': 10, 'alpha_matting_erode_size': 10, 'alpha_matting_base_size': 1000, 'workernodes': 1, 'gpubatchsize': 2, 'framerate': -1, 'framelimit': -1, 'mattekey': False, 'transparentvideo': False, 'transparentvideoovervideo': False, 'transparentvideooverimage': False, 'transparentgif': False, 'transparentgifwithbackground': False, 'input': RESOURCES_PATH + 'images/test_image.jpg', 'backgroundimage': RESOURCES_PATH + 'images/test_image.jpg', 'backgroundvideo': RESOURCES_PATH + 'videos/test_video.mp4', 'output': 'path/to/output/image.png', 'input_folder': None, 'output_folder': None}

    def is_video_file(filename):
        return filename.lower().endswith(('.mp4', '.mov', '.webm', '.ogg', '.gif'))

    def is_image_file(filename):
        return filename.lower().endswith(('.jpg', '.jpeg', '.png'))
    if args['input_folder']:
        input_folder = os.path.abspath(args['input_folder'])
        output_folder = os.path.abspath(args['output_folder'] or input_folder)
        os.makedirs(output_folder, exist_ok=True)
        files = [f for f in os.listdir(input_folder) if is_video_file(f) or is_image_file(f)]
        for f in files:
            input_path = os.path.join(input_folder, f)
            output_path = os.path.join(output_folder, f'output_{f}')
            if is_video_file(f):
                if args['mattekey']:
                    utilities.matte_key(output_path, input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
                elif args['transparentvideo']:
                    utilities.transparentvideo(output_path, input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
                elif args['transparentvideoovervideo']:
                    utilities.transparentvideoovervideo(output_path, os.path.abspath(args['backgroundvideo']), input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
                elif args['transparentvideooverimage']:
                    utilities.transparentvideooverimage(output_path, os.path.abspath(args['backgroundimage']), input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
                elif args['transparentgif']:
                    utilities.transparentgif(output_path, input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
                elif args['transparentgifwithbackground']:
                    utilities.transparentgifwithbackground(output_path, os.path.abspath(args['backgroundimage']), input_path, worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
            elif is_image_file(f):
                with open(input_path, 'rb') as i, open(output_path, 'wb') as o:
                    r = lambda i: i.buffer.read() if hasattr(i, 'buffer') else i.read()
                    w = lambda o, data: o.buffer.write(data) if hasattr(o, 'buffer') else o.write(data)
                    w(o, exe.run('remove', data=r(i), model_name=args['model'], alpha_matting=args['alpha_matting'], alpha_matting_foreground_threshold=args['alpha_matting_foreground_threshold'], alpha_matting_background_threshold=args['alpha_matting_background_threshold'], alpha_matting_erode_structure_size=args['alpha_matting_erode_size'], alpha_matting_base_size=args['alpha_matting_base_size']))
        return
    ext = os.path.splitext(args['input'])[1].lower()
    if ext in ['.mp4', '.mov', '.webm', '.ogg', '.gif']:
        if args['mattekey']:
            utilities.matte_key(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
        elif args['transparentvideo']:
            utilities.transparentvideo(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
        elif args['transparentvideoovervideo']:
            utilities.transparentvideoovervideo(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['backgroundvideo']), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
        elif args['transparentvideooverimage']:
            utilities.transparentvideooverimage(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['backgroundimage']), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
        elif args['transparentgif']:
            utilities.transparentgif(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
        elif args['transparentgifwithbackground']:
            utilities.transparentgifwithbackground(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), os.path.abspath(args['backgroundimage']), os.path.abspath(args['input']), worker_nodes=args['workernodes'], gpu_batchsize=args['gpubatchsize'], model_name=args['model'], frame_limit=args['framelimit'], framerate=args['framerate'])
    elif ext in ['.jpg', '.jpeg', '.png']:
        r = lambda i: i.buffer.read() if hasattr(i, 'buffer') else i.read()
        w = lambda o, data: o.buffer.write(data) if hasattr(o, 'buffer') else o.write(data)
        w(open(os.path.join(FILE_RECORD_PATH, os.path.basename(args['output'])), 'wb'), exe.run('remove', data=r(open(args['input'], 'rb')), model_name=args['model'], alpha_matting=args['alpha_matting'], alpha_matting_foreground_threshold=args['alpha_matting_foreground_threshold'], alpha_matting_background_threshold=args['alpha_matting_background_threshold'], alpha_matting_erode_structure_size=args['alpha_matting_erode_size'], alpha_matting_base_size=args['alpha_matting_base_size']))
    else:
        print(f'❌ Unsupported file type: {ext}')
        exit(1)
execute_background_removal()