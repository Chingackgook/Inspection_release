import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.mmdetection import ENV_DIR
from Inspection.adapters.custom_adapters.mmdetection import *
exe = Executor('mmdetection', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import ast
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
'Image Demo.\n\nThis script adopts a new inference class, currently supports image path,\nnp.array and folder input formats, and will support video and webcam\nin the future.\n\nExample:\n    Save visualizations and predictions results::\n\n        python demo/image_demo.py demo/demo.jpg rtmdet-s\n        ...\n'
mock_args = {'inputs': os.path.join(ENV_DIR, 'demo/demo.jpg'), 'model': 'rtmdet-s', 'weights': None, 'out_dir': os.path.join(FILE_RECORD_PATH, 'outputs'), 'texts': None, 'device': 'cuda:0', 'pred_score_thr': 0.3, 'batch_size': 1, 'show': False, 'no_save_vis': False, 'no_save_pred': False, 'print_result': False, 'palette': 'none', 'custom_entities': False, 'chunked_size': -1, 'tokens_positive': None}

def parse_args():
    call_args = mock_args.copy()
    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''
    if call_args['model'] and call_args['model'].endswith('.pth'):
        call_args['weights'] = call_args['model']
        call_args['model'] = None
    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]
    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(call_args['tokens_positive'])
    return call_args
call_args = parse_args()
exe.create_interface_objects(model=call_args['model'], weights=call_args['weights'], device=call_args['device'], palette=call_args['palette'])
inputs = call_args['inputs']
batch_size = call_args['batch_size']
preds = exe.run('preprocess', inputs=inputs, batch_size=batch_size)
chunked_size = call_args.pop('chunked_size')
exe.adapter.det_inferencer.model.test_cfg.chunked_size = chunked_size
visualization = exe.run('visualize', inputs=inputs, preds=preds, return_vis=True, show=call_args['show'], wait_time=0, draw_pred=True, pred_score_thr=call_args['pred_score_thr'], no_save_vis=call_args['no_save_vis'], img_out_dir=call_args['out_dir'])
results = exe.run('postprocess', preds=preds, visualization=visualization, return_datasamples=False, print_result=call_args['print_result'], no_save_pred=call_args['no_save_pred'], pred_out_dir=call_args['out_dir'])
if call_args['out_dir'] != '' and (not (call_args['no_save_vis'] and call_args['no_save_pred'])):
    print_log(f"results have been saved at {call_args['out_dir']}")