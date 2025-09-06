from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.fawkes import *
import glob
import logging
import os
import sys
import tensorflow as tf
import numpy as np
from fawkes.differentiator import FawkesMaskGeneration
from fawkes.utils import init_gpu
from fawkes.utils import dump_image
from fawkes.utils import reverse_process_cloaked
from fawkes.utils import Faces
from fawkes.utils import filter_image_paths
from fawkes.utils import load_extractor
from fawkes.align_face import aligner
import signal
exe = Executor('fawkes', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/fawkes/fawkes/protection.py'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['KMP_AFFINITY'] = 'noverbose'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)


def generate_cloak_images(protector, image_X, target_emb=None):
    cloaked_image_X = protector.compute(image_X, target_emb)
    return cloaked_image_X


IMG_SIZE = 112
PREPROCESS = 'raw'


def execute_fawkes():
    directory = 'imgs/'
    gpu = '0'
    mode = 'low'
    feature_extractor = 'arcface_extractor_0'
    th = 0.01
    max_step = 1000
    sd = 1000000.0
    lr = 2
    batch_size = 1
    separate_target = False
    no_align = False
    debug = False
    format = 'png'
    assert format in ['png', 'jpg', 'jpeg']
    if format == 'jpg':
        format = 'jpeg'
    image_paths = glob.glob(os.path.join(directory, '*'))
    image_paths = [path for path in image_paths if '_cloaked' not in path.
        split('/')[-1]]
    protector = exe.create_interface_objects(interface_class_name='Fawkes',
        feature_extractor=feature_extractor, gpu=gpu, batch_size=batch_size,
        mode=mode)
    var = exe.run('run_protection', image_paths=image_paths, th=th, sd=sd,
        lr=lr, max_step=max_step, batch_size=batch_size, format=format,
        separate_target=separate_target, debug=debug, no_align=no_align)


execute_fawkes()
