from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.mmpose import *
import sys
exe = Executor('mmpose', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/mmpose/demo/inferencer_demo.py'
from argparse import ArgumentParser
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases
filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True), rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True))
inputs = RESOURCES_PATH + 'videos/test_video.mp4'
# add
pose2d = '/mnt/autor_name/haoTingDeWenJianJia/mmpose/configs/body_2d_keypoint/associative_embedding/coco/ae_hrnet-w32_8xb24-300e_coco-512x512.py'
# end add


# origin code
# pose2d = 'yoloxpose'
pose2d_weights = None
pose3d = None
pose3d_weights = None
det_model = None
det_weights = None
det_cat_ids = [0]
scope = 'mmpose'
device = None
show_progress = False
show_alias = False
call_args = {'show': False, 'draw_bbox': False, 'draw_heatmap': False, 'bbox_thr': filter_args['bbox_thr'], 'nms_thr': filter_args['nms_thr'], 'pose_based_nms': filter_args['pose_based_nms'], 'kpt_thr': 0.3, 'tracking_thr': 0.3, 'use_oks_tracking': False, 'disable_norm_pose_2d': False, 'disable_rebase_keypoint': False, 'num_instances': 1, 'radius': 3, 'thickness': 1, 'skeleton_style': 'mmpose', 'black_background': False, 'vis_out_dir': '', 'pred_out_dir': ''}
init_args = {'pose2d': pose2d, 'pose2d_weights': pose2d_weights, 'scope': scope, 'device': device, 'det_model': det_model, 'det_weights': det_weights, 'det_cat_ids': det_cat_ids, 'pose3d': pose3d, 'pose3d_weights': pose3d_weights, 'show_progress': show_progress}

def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f"{'ALIAS'.ljust(max_alias_length + 2)}MODEL_NAME")
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length + 2)}{model_aliases[alias]}')

def run_inference():
    """Run the inference process."""
    if show_alias:
        model_aliases = get_model_aliases(init_args['scope'])
        display_model_aliases(model_aliases)
    else:
        inferencer = exe.create_interface_objects(interface_class_name='MMPoseInferencer', pose2d=pose2d, pose2d_weights=pose2d_weights, pose3d=pose3d, pose3d_weights=pose3d_weights, device=device, scope=scope, det_model=det_model, det_weights=det_weights, det_cat_ids=det_cat_ids, show_progress=show_progress)
        for _ in exe.run('__call__',inputs=inputs, return_datasamples=False, batch_size=1, out_dir=None, **call_args):
            pass
run_inference()