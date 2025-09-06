from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.PaddleDetection import ENV_DIR
from Inspection.adapters.custom_adapters.PaddleDetection import *
exe = Executor('PaddleDetection', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

import os
import sys
import warnings
import glob
import ast
import paddle
from ppdet.core.workspace import create
from ppdet.core.workspace import load_config
from ppdet.core.workspace import merge_config
from ppdet.engine import Trainer
from ppdet.engine import Trainer_ARSL
from ppdet.utils.check import check_gpu
from ppdet.utils.check import check_npu
from ppdet.utils.check import check_xpu
from ppdet.utils.check import check_mlu
from ppdet.utils.check import check_gcu
from ppdet.utils.check import check_version
from ppdet.utils.check import check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.cli import merge_args
from ppdet.slim import build_slim_model
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')
FLAGS = {'infer_dir': os.path.join(ENV_DIR, 'infer_dir'), 
         'config': os.path.join(ENV_DIR, 'configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml'),
         'opt' : {'weights': 'https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams'},
         'infer_list': None, 'infer_img': None, 'output_dir': os.path.join(FILE_RECORD_PATH, 'output'), 'draw_threshold': 0.5, 'save_threshold': 0.5, 'slim_config': None, 'use_vdl': False, 'do_eval': False, 'vdl_log_dir': os.path.join(FILE_RECORD_PATH, 'vdl_log_dir/image'), 'save_results': False, 'slice_infer': False, 'slice_size': [640, 640], 'overlap_ratio': [0.25, 0.25], 'combine_method': 'nms', 'match_threshold': 0.6, 'match_metric': 'ios', 'visualize': True, 'rtn_im_file': False}

def parse_args():
    return FLAGS

def get_test_images(infer_dir, infer_img, infer_list=None):
    assert infer_img is not None or infer_dir is not None, '--infer_img or --infer_dir should be set'
    assert infer_img is None or os.path.isfile(infer_img), '{} is not a file'.format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), '{} is not a directory'.format(infer_dir)
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]
    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), 'infer_dir {} is not a directory'.format(infer_dir)
    if infer_list:
        assert os.path.isfile(infer_list), f'infer_list {infer_list} is not a valid file path.'
        with open(infer_list, 'r') as f:
            lines = f.readlines()
        for line in lines:
            images.update([os.path.join(infer_dir, line.strip())])
    else:
        exts = ['jpg', 'jpeg', 'png', 'bmp']
        exts += [ext.upper() for ext in exts]
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)
    assert len(images) > 0, 'no image found in {}'.format(infer_dir)
    logger.info('Found {} inference images in total.'.format(len(images)))
    return images

def run(FLAGS, cfg):
    exe.create_interface_objects(cfg=cfg, mode='test')
    ssod_method = cfg.get('ssod_method', None)
    if ssod_method == 'ARSL':
        exe.run('load_weights', weights=cfg['weights'], ARSL_eval=True)
    else:
        exe.run('load_weights', weights=cfg['weights'])
    if FLAGS['do_eval']:
        dataset = create('TestDataset')()
        images = dataset.get_images()
    else:
        images = get_test_images(FLAGS['infer_dir'], FLAGS['infer_img'], FLAGS['infer_list'])
    if FLAGS['slice_infer']:
        exe.run('slice_predict', images=images, slice_size=FLAGS['slice_size'], overlap_ratio=FLAGS['overlap_ratio'], combine_method=FLAGS['combine_method'], match_threshold=FLAGS['match_threshold'], match_metric=FLAGS['match_metric'], draw_threshold=FLAGS['draw_threshold'], output_dir=FLAGS['output_dir'], save_results=FLAGS['save_results'], visualize=FLAGS['visualize'])
    else:
        exe.run('predict', images=images, draw_threshold=FLAGS['draw_threshold'], output_dir=FLAGS['output_dir'], save_results=FLAGS['save_results'], visualize=FLAGS['visualize'], save_threshold=FLAGS['save_threshold'], do_eval=FLAGS['do_eval'])
FLAGS = parse_args()
cfg = load_config(FLAGS['config'])
merge_args(cfg, FLAGS)
merge_config(FLAGS['opt'])
if 'use_npu' not in cfg:
    cfg['use_npu'] = False
if 'use_xpu' not in cfg:
    cfg['use_xpu'] = False
if 'use_gpu' not in cfg:
    cfg['use_gpu'] = False
if 'use_mlu' not in cfg:
    cfg['use_mlu'] = False
if 'use_gcu' not in cfg:
    cfg['use_gcu'] = False
if cfg['use_gpu']:
    place = paddle.set_device('gpu')
elif cfg['use_npu']:
    place = paddle.set_device('npu')
elif cfg['use_xpu']:
    place = paddle.set_device('xpu')
elif cfg['use_mlu']:
    place = paddle.set_device('mlu')
elif cfg['use_gcu']:
    place = paddle.set_device('gcu')
else:
    place = paddle.set_device('cpu')
if FLAGS['slim_config']:
    cfg = build_slim_model(cfg, FLAGS['slim_config'], mode='test')
check_config(cfg)
check_gpu(cfg['use_gpu'])
check_npu(cfg['use_npu'])
check_xpu(cfg['use_xpu'])
check_mlu(cfg['use_mlu'])
check_gcu(cfg['use_gcu'])
check_version()
run(FLAGS, cfg)