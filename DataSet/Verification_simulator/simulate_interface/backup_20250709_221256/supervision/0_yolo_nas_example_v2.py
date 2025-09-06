from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.supervision import ENV_DIR
from Inspection.adapters.custom_adapters.supervision import *
exe = Executor('supervision','simulation')
FILE_RECORD_PATH = exe.now_record_path

# 可能需要手动修改的部分：
source_video_path = ENV_DIR + 'source/video.mp4'  # 输入视频路径
target_video_path = FILE_RECORD_PATH + 'output/video.mp4'  # 输出视频路径
confidence_threshold = 0.3  # 置信度阈值
iou_threshold = 0.7  # IOU阈值
# end

import argparse
from collections import defaultdict
from collections import deque
import cv2
import numpy as np
from super_gradients.common.object_names import Models
from super_gradients.training import models
import supervision as sv

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])
TARGET_WIDTH = 25
TARGET_HEIGHT = 250
TARGET = np.array([[0, 0], [TARGET_WIDTH - 1, 0], [TARGET_WIDTH - 1, TARGET_HEIGHT - 1], [0, TARGET_HEIGHT - 1]])

class ViewTransformer:

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def main(source_video_path, target_video_path, confidence_threshold=0.3, iou_threshold=0.7):
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    model = models.get(Models.YOLO_NAS_L, pretrained_weights='coco')
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence_threshold)
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=video_info.fps * 2, position=sv.Position.BOTTOM_CENTER)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
    coordinates = defaultdict(lambda : deque(maxlen=video_info.fps))
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model.predict(frame)
            # 修改前：detections = exe.create_interface_objects('Detections', result=result)
            # 修改后
            detections = sv.Detections.from_yolo_nas(result)
            exe.adapter.detections_instance = detections
            # 结束
            detections = detections[polygon_zone.trigger(detections)]
            detections = exe.run('with_nms', threshold=iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)
            for (tracker_id, [_, y]) in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f'#{tracker_id}')
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f'#{tracker_id} {int(speed)} km/h')
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            sink.write_frame(annotated_frame)
            cv2.imshow('frame', annotated_frame)
            if cv2.waitKey(1) & 255 == ord('q'):
                break
        cv2.destroyAllWindows()

# 直接运行主逻辑
main(source_video_path, target_video_path, confidence_threshold, iou_threshold)
