# face 
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'face/'
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/face')
# 以上是自动生成的代码，请勿修改



import face_recognition
from abc import ABC, abstractmethod
from typing import Any, Dict, List

class ExecutionResult:
    def __init__(self):
        self.fuc_name = ''
        self.is_success = False
        self.fail_reason = ''
        self.is_file = False
        self.file_path = ''
        self.except_data = None
        self.interface_return = None

    def set_result(
        self, 
        fuc_name: str,
        is_success: bool,
        fail_reason: str,
        is_file: bool, 
        file_path: str, 
        except_data: Any, 
        interface_return: Any,
    ):
        self.fuc_name = fuc_name
        self.is_success = is_success
        self.fail_reason = fail_reason
        self.is_file = is_file
        self.file_path = file_path
        self.interface_return = interface_return
        self.except_data = except_data

class BaseAdapter(ABC):
    def __init__(self):
        self.result: ExecutionResult = ExecutionResult()
    
    @abstractmethod
    def create_interface_objects(self, **kwargs):
        pass

    @abstractmethod
    def run(self, name: str, **kwargs):
        pass

class CustomAdapter(BaseAdapter):
    def create_interface_objects(self, **kwargs):
        # 这里可以实现模型的加载逻辑
        pass

    def run(self, name: str, **kwargs):
        try:
            if name == 'face_distance':
                self.result.interface_return = face_recognition.face_distance(**kwargs)
            elif name == 'load_image_file':
                self.result.interface_return = face_recognition.load_image_file(**kwargs)
            elif name == 'face_locations':
                self.result.interface_return = face_recognition.face_locations(**kwargs)
            elif name == 'face_landmarks':
                self.result.interface_return = face_recognition.face_landmarks(**kwargs)
            elif name == 'face_encodings':
                self.result.interface_return = face_recognition.face_encodings(**kwargs)
            elif name == 'compare_faces':
                self.result.interface_return = face_recognition.compare_faces(**kwargs)
            else:
                raise ValueError("Invalid function name provided.")
            
            self.result.set_result(
                fuc_name=name,
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=self.result.interface_return
            )
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

if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)