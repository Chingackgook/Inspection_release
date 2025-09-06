from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.mmdetection import *
exe = Executor('mmdetection', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/mmdetection/demo/image_demo.py'

# Import the existing package
import ast
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
# end

inputs = 'demo/demo.jpg'
# add
model = 'rtmdet-s'
weights = None
# add end
# orign code:
# model = 'configs/rtmdet/rtmdet_s_8xb32-300e_coco.py'
# weights = 'rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth'
out_dir = FILE_RECORD_PATH  # Replaced with FILE_RECORD_PATH
texts = None
device = 'cuda:0'
pred_score_thr = 0.3
batch_size = 1
show = False
no_save_vis = False
no_save_pred = False
print_result = False
palette = 'none'
custom_entities = False
chunked_size = -1
tokens_positive = None
call_args = {
    'inputs': inputs,
    'model': model,
    'weights': weights,
    'out_dir': out_dir,
    'texts': texts,
    'device': device,
    'pred_score_thr': pred_score_thr,
    'batch_size': batch_size,
    'show': show,
    'no_save_vis': no_save_vis,
    'no_save_pred': no_save_pred,
    'print_result': print_result,
    'palette': palette,
    'custom_entities': custom_entities,
    'chunked_size': chunked_size,
    'tokens_positive': tokens_positive
}

if call_args['no_save_vis'] and call_args['no_save_pred']:
    call_args['out_dir'] = ''
if call_args['model'].endswith('.pth'):
    print_log(
        'The model is a weight file, automatically assign the model to --weights'
    )
    call_args['weights'] = call_args['model']
    call_args['model'] = None
if call_args['texts'] is not None:
    if call_args['texts'].startswith('$:'):
        dataset_name = call_args['texts'][3:].strip()
        class_names = get_classes(dataset_name)
        call_args['texts'] = [tuple(class_names)]
if call_args['tokens_positive'] is not None:
    call_args['tokens_positive'] = ast.literal_eval(call_args['tokens_positive'])

init_kws = ['model', 'weights', 'device', 'palette']
init_args = {}
for init_kw in init_kws:
    init_args[init_kw] = call_args.pop(init_kw)

inferencer = exe.create_interface_objects(
    interface_class_name='DetInferencer',
    model=init_args['model'],
    weights=init_args['weights'],
    device=init_args['device'],
    palette=init_args['palette']
)

chunked_size = call_args.pop('chunked_size')
inferencer.model.test_cfg.chunked_size = chunked_size
_ = exe.run('__call__', **call_args)

if call_args['out_dir'] != '' and not (call_args['no_save_vis'] and call_args['no_save_pred']):
    print_log(f"results have been saved at {call_args['out_dir']}")
