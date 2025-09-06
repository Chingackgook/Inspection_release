# deep_live_cam 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'deep_live_cam/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/Deep-Live-Cam')
# 以上是自动生成的代码，请勿修改
from modules.processors.frame.face_swapper import *

from abc import ABC
from typing import Any, Dict, List

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        # 不需要加载任何模型以及初始化
        self.result.set_result(
            fuc_name='create_interface_objects',
            is_success=True,
            fail_reason='',
            is_file=False,
            file_path='',
            except_data=None,
            interface_return=None
        )

    def run(self, name: str, **kwargs):
        try:
            if name == 'pre_check':
                result = pre_check()
            elif name == 'pre_start':
                result = pre_start()
            elif name == 'get_face_swapper':
                result = get_face_swapper()
            elif name == 'swap_face':
                result = swap_face(**kwargs)
            elif name == 'process_frame':
                result = process_frame(**kwargs)
            elif name == 'process_frame_v2':
                result = process_frame_v2(**kwargs)
            elif name == 'process_frames':
                result = process_frames(**kwargs)
            elif name == 'process_image':
                result = process_image(**kwargs)
            elif name == 'process_video':
                result = process_video(**kwargs)
            elif name == 'create_lower_mouth_mask':
                result = create_lower_mouth_mask(**kwargs)
            elif name == 'draw_mouth_mask_visualization':
                result = draw_mouth_mask_visualization(**kwargs)
            elif name == 'apply_mouth_area':
                result = apply_mouth_area(**kwargs)
            elif name == 'create_face_mask':
                result = create_face_mask(**kwargs)
            elif name == 'apply_color_transfer':
                result = apply_color_transfer(**kwargs)
            else:
                raise ValueError(f"Unknown function name: {name}")
        except Exception as e:
            self.result.set_result(
                fuc_name=name,
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )
            return
        self.result.set_result(
            fuc_name=name,
            is_success=True,
            fail_reason='',
            is_file=False,
            file_path='',
            except_data=None,
            interface_return=result
        )

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 addtional_data
# 该属性用于存储函数名，等
adapter_addtional_data = {}
adapter_addtional_data['functions'] = []
adapter_addtional_data['functions'].append('pre_check')
adapter_addtional_data['functions'].append('pre_start')
adapter_addtional_data['functions'].append('get_face_swapper')
adapter_addtional_data['functions'].append('swap_face')
adapter_addtional_data['functions'].append('process_frame')
adapter_addtional_data['functions'].append('process_frame_v2')
adapter_addtional_data['functions'].append('process_frames')
adapter_addtional_data['functions'].append('process_image')
adapter_addtional_data['functions'].append('process_video')
adapter_addtional_data['functions'].append('create_lower_mouth_mask')
adapter_addtional_data['functions'].append('draw_mouth_mask_visualization')
adapter_addtional_data['functions'].append('apply_mouth_area')
adapter_addtional_data['functions'].append('create_face_mask')
adapter_addtional_data['functions'].append('apply_color_transfer')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.addtional_data = adapter_addtional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
