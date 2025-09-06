# supervision 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'supervision/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.insert(0, '/mnt/autor_name/haoTingDeWenJianJia/supervision')
os.chdir('/mnt/autor_name/haoTingDeWenJianJia/supervision')

# you can add your custom imports here
from supervision.detection.core import *

# DeadCodeFront end

class CustomAdapter(BaseAdapter):

    def create_interface_objects(self, interface_class_name: str = "", **kwargs):
        self.class1_obj = None  # Interface object for Detections

        try:
            if interface_class_name == 'Detections':
                # Create interface object
                self.class1_obj = Detections(**kwargs)
                self.result.interface_return = self.class1_obj
            elif interface_class_name == '':
                # If omitted, create a default interface object
                self.class1_obj = Detections(**kwargs)
                self.result.interface_return = self.class1_obj

            self.result.is_success = True
            self.result.fail_reason = ''
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_file = False
            self.result.file_path = ''

        except Exception as e:
            self.result.fuc_name = 'create_interface_objects'
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to create interface object: {e}")

    def run(self, dispatch_key: str, **kwargs):
        try:
            if dispatch_key == '__len__':
                self.result.interface_return = self.class1_obj.__len__(**kwargs)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == '__iter__':
                self.result.interface_return = list(self.class1_obj.__iter__(**kwargs))
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == '__eq__':
                other = kwargs.get('other')
                self.result.interface_return = self.class1_obj.__eq__(other)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'get_anchors_coordinates':
                anchor = kwargs.get('anchor')
                self.result.interface_return = self.class1_obj.get_anchors_coordinates(anchor)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'is_empty':
                self.result.interface_return = self.class1_obj.is_empty()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'area':
                self.result.interface_return = self.class1_obj.area
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'box_area':
                self.result.interface_return = self.class1_obj.box_area
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'with_nms':
                threshold = kwargs.get('threshold', 0.5)
                class_agnostic = kwargs.get('class_agnostic', False)
                self.result.interface_return = self.class1_obj.with_nms(threshold, class_agnostic)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'with_nmm':
                threshold = kwargs.get('threshold', 0.5)
                class_agnostic = kwargs.get('class_agnostic', False)
                self.result.interface_return = self.class1_obj.with_nmm(threshold, class_agnostic)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_yolov5':
                yolov5_results = kwargs.get('yolov5_results')
                self.result.interface_return = Detections.from_yolov5(yolov5_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_ultralytics':
                ultralytics_results = kwargs.get('ultralytics_results')
                self.result.interface_return = Detections.from_ultralytics(ultralytics_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_yolo_nas':
                yolo_nas_results = kwargs.get('yolo_nas_results')
                self.result.interface_return = Detections.from_yolo_nas(yolo_nas_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_tensorflow':
                tensorflow_results = kwargs.get('tensorflow_results')
                resolution_wh = kwargs.get('resolution_wh')
                self.result.interface_return = Detections.from_tensorflow(tensorflow_results, resolution_wh)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_deepsparse':
                deepsparse_results = kwargs.get('deepsparse_results')
                self.result.interface_return = Detections.from_deepsparse(deepsparse_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_mmdetection':
                mmdet_results = kwargs.get('mmdet_results')
                self.result.interface_return = Detections.from_mmdetection(mmdet_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_transformers':
                transformers_results = kwargs.get('transformers_results')
                id2label = kwargs.get('id2label')
                self.result.interface_return = Detections.from_transformers(transformers_results, id2label)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_detectron2':
                detectron2_results = kwargs.get('detectron2_results')
                self.result.interface_return = Detections.from_detectron2(detectron2_results)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_inference':
                roboflow_result = kwargs.get('roboflow_result')
                self.result.interface_return = Detections.from_inference(roboflow_result)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_sam':
                sam_result = kwargs.get('sam_result')
                self.result.interface_return = Detections.from_sam(sam_result)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_azure_analyze_image':
                azure_result = kwargs.get('azure_result')
                class_map = kwargs.get('class_map')
                self.result.interface_return = Detections.from_azure_analyze_image(azure_result, class_map)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'from_paddledet':
                paddledet_result = kwargs.get('paddledet_result')
                self.result.interface_return = Detections.from_paddledet(paddledet_result)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'empty':
                self.result.interface_return = Detections.empty()
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            elif dispatch_key == 'merge':
                detections_list = kwargs.get('detections_list')
                self.result.interface_return = Detections.merge(detections_list)
                self.result.is_success = True
                self.result.fail_reason = ''
                self.result.fuc_name = dispatch_key
            else:
                raise ValueError(f"Unknown interface method: {dispatch_key}")

        except Exception as e:
            self.result.fuc_name = dispatch_key
            self.result.is_success = False
            self.result.fail_reason = str(e)
            self.result.interface_return = None
            print(f"[INS_ERROR] Failed to execute interface {dispatch_key}: {e}")

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
