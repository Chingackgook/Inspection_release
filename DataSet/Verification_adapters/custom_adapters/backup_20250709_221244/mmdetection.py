# mmdetection 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'mmdetection/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/mmdetection')

# 可以在此位置后添加导包部分代码
import ast
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
from typing import Optional,Union,List, Dict, Any
from mmengine.infer.infer import BaseInferencer, ModelType

# DeadCodeFront end

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.det_inferencer = None

    def create_interface_objects(self, model: Optional[Union[ModelType, str]] = None, weights: Optional[str] = None, device: Optional[str] = None, scope: Optional[str] = 'mmdet', palette: str = 'none', show_progress: bool = True):
        try:
            self.det_inferencer = DetInferencer(model=model, weights=weights, device=device, scope=scope, palette=palette, show_progress=show_progress)
            self.result.set_result('create_interface_objects', True, '', False, '', None, None)
        except Exception as e:
            self.result.set_result('create_interface_objects', False, str(e), False, '', None, None)

    def run(self, name: str, **kwargs):
        try:
            if name == 'preprocess':
                result = self.det_inferencer.preprocess(**kwargs)
                self.result.set_result('preprocess', True, '', False, '', result, result)
            elif name == 'visualize':
                result = self.det_inferencer.visualize(**kwargs)
                self.result.set_result('visualize', True, '', False, '', result, result)
            elif name == 'postprocess':
                result = self.det_inferencer.postprocess(**kwargs)
                self.result.set_result('postprocess', True, '', False, '', result, result)
            elif name == 'pred2dict':
                result = self.det_inferencer.pred2dict(**kwargs)
                self.result.set_result('pred2dict', True, '', False, '', result, result)
            elif name == '_load_weights_to_model':
                self.det_inferencer._load_weights_to_model(**kwargs)
                self.result.set_result('_load_weights_to_model', True, '', False, '', None, None)
            elif name == '_init_pipeline':
                result = self.det_inferencer._init_pipeline(**kwargs)
                self.result.set_result('_init_pipeline', True, '', False, '', result, result)
            elif name == '_get_transform_idx':
                result = self.det_inferencer._get_transform_idx(**kwargs)
                self.result.set_result('_get_transform_idx', True, '', False, '', result, result)
            elif name == '_init_visualizer':
                result = self.det_inferencer._init_visualizer(**kwargs)
                self.result.set_result('_init_visualizer', True, '', False, '', result, result)
            elif name == '_inputs_to_list':
                result = self.det_inferencer._inputs_to_list(**kwargs)
                self.result.set_result('_inputs_to_list', True, '', False, '', result, result)
            elif name == '_get_chunk_data':
                result = self.det_inferencer._get_chunk_data(**kwargs)
                self.result.set_result('_get_chunk_data', True, '', False, '', result, result)
            elif name == '_get_transform_idx':
                result = self.det_inferencer._get_transform_idx(**kwargs)
                self.result.set_result('_get_transform_idx', True, '', False, '', result, result)
            else:
                self.result.set_result(name, False, 'Method not found', False, '', None, None)
        except Exception as e:
            self.result.set_result(name, False, str(e), False, '', None, None)

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('preprocess')
adapter_additional_data['functions'].append('visualize')
adapter_additional_data['functions'].append('postprocess')
adapter_additional_data['functions'].append('pred2dict')
adapter_additional_data['functions'].append('_load_weights_to_model')
adapter_additional_data['functions'].append('_init_pipeline')
adapter_additional_data['functions'].append('_get_transform_idx')
adapter_additional_data['functions'].append('_init_visualizer')
adapter_additional_data['functions'].append('_inputs_to_list')
adapter_additional_data['functions'].append('_get_chunk_data')
adapter_additional_data['functions'].append('_get_transform_idx')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
