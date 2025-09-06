# gfpgan 
from Inspection import ENV_BASE
ENV_DIR = ENV_BASE + 'gfpgan/'
from Inspection.adapters import BaseAdapter
from Inspection.adapters import ExecutionResult
import sys
import os
sys.path.append('/mnt/autor_name/haoTingDeWenJianJia/GFPGAN')
# 以上是自动生成的代码，请勿修改

from gfpgan import GFPGANer
from basicsr.utils.download_util import load_file_from_url
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils import img2tensor, tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from torchvision.transforms.functional import normalize
from abc import ABC
from typing import Any
import torch
import numpy as np

class CustomAdapter(BaseAdapter):
    def __init__(self):
        super().__init__()
        self.gfpganer = None
        self.face_restore_helper = None

    def create_interface_objects(self, **kwargs):
        model_path = kwargs.get('model_path', None)
        upscale = kwargs.get('upscale', 2)
        arch = kwargs.get('arch', 'clean')
        channel_multiplier = kwargs.get('channel_multiplier', 2)
        bg_upsampler = kwargs.get('bg_upsampler', None)
        device = kwargs.get('device', None)

        try:
            self.gfpganer = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch=arch,
                channel_multiplier=channel_multiplier,
                bg_upsampler=bg_upsampler,
                device=device
            )
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=True,
                fail_reason='',
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )
        except Exception as e:
            self.result.set_result(
                fuc_name='create_interface_objects',
                is_success=False,
                fail_reason=str(e),
                is_file=False,
                file_path='',
                except_data=None,
                interface_return=None
            )

    def run(self, name: str, **kwargs):
        try:
            if name == 'enhance':
                img = kwargs.get('img')
                has_aligned = kwargs.get('has_aligned', False)
                only_center_face = kwargs.get('only_center_face', False)
                paste_back = kwargs.get('paste_back', True)
                weight = kwargs.get('weight', 0.5)

                cropped_faces, restored_faces, restored_img = self.gfpganer.enhance(
                    img,
                    has_aligned=has_aligned,
                    only_center_face=only_center_face,
                    paste_back=paste_back,
                    weight=weight
                )
                self.result.set_result(
                    fuc_name='enhance',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=(cropped_faces, restored_faces, restored_img),
                    interface_return=(cropped_faces, restored_faces, restored_img)
                )
            elif name == 'load_file_from_url':
                url = kwargs.get('url')
                model_dir = kwargs.get('model_dir')
                progress = kwargs.get('progress', True)
                file_name = kwargs.get('file_name', None)

                local_path = load_file_from_url(url, model_dir, progress, file_name)
                self.result.set_result(
                    fuc_name='load_file_from_url',
                    is_success=True,
                    fail_reason='',
                    is_file=True,
                    file_path=local_path,
                    except_data=None,
                    interface_return=local_path
                )
            elif name == 'img2tensor':
                img = kwargs.get('img')
                bgr2rgb = kwargs.get('bgr2rgb', True)
                float32 = kwargs.get('float32', True)

                tensor = img2tensor(img, bgr2rgb, float32)
                self.result.set_result(
                    fuc_name='img2tensor',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=tensor,
                    interface_return=tensor
                )
            elif name == 'normalize':
                tensor = kwargs.get('tensor')
                mean = kwargs.get('mean', (0.5, 0.5, 0.5))
                std = kwargs.get('std', (0.5, 0.5, 0.5))
                inplace = kwargs.get('inplace', False)

                normalize(tensor, mean, std, inplace)
                self.result.set_result(
                    fuc_name='normalize',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'tensor2img':
                tensor = kwargs.get('tensor')
                rgb2bgr = kwargs.get('rgb2bgr', True)
                min_max = kwargs.get('min_max', (-1, 1))

                img = tensor2img(tensor, rgb2bgr, min_max)
                self.result.set_result(
                    fuc_name='tensor2img',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=img,
                    interface_return=img
                )
            elif name == 'FaceRestoreHelper':
                upscale = kwargs.get('upscale', 2)
                face_size = kwargs.get('face_size', 512)
                crop_ratio = kwargs.get('crop_ratio', (1, 1))
                det_model = kwargs.get('det_model', 'retinaface_resnet50')
                save_ext = kwargs.get('save_ext', 'png')
                use_parse = kwargs.get('use_parse', True)
                device = kwargs.get('device', None)
                model_rootpath = kwargs.get('model_rootpath', '')

                self.face_restore_helper = FaceRestoreHelper(
                    upscale=upscale,
                    face_size=face_size,
                    crop_ratio=crop_ratio,
                    det_model=det_model,
                    save_ext=save_ext,
                    use_parse=use_parse,
                    device=device,
                    model_rootpath=model_rootpath
                )
                self.result.set_result(
                    fuc_name='FaceRestoreHelper',
                    is_success=True,
                    fail_reason='',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
                )
            elif name == 'clean_all':
                if self.face_restore_helper:
                    self.face_restore_helper.clean_all()
                    self.result.set_result(
                        fuc_name='clean_all',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='clean_all',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'read_image':
                img = kwargs.get('img')
                if self.face_restore_helper:
                    self.face_restore_helper.read_image(img)
                    self.result.set_result(
                        fuc_name='read_image',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='read_image',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'get_face_landmarks_5':
                only_center_face = kwargs.get('only_center_face', False)
                eye_dist_threshold = kwargs.get('eye_dist_threshold', 5)
                if self.face_restore_helper:
                    self.face_restore_helper.get_face_landmarks_5(
                        only_center_face=only_center_face,
                        eye_dist_threshold=eye_dist_threshold
                    )
                    self.result.set_result(
                        fuc_name='get_face_landmarks_5',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='get_face_landmarks_5',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'align_warp_face':
                if self.face_restore_helper:
                    self.face_restore_helper.align_warp_face()
                    self.result.set_result(
                        fuc_name='align_warp_face',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='align_warp_face',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'add_restored_face':
                restored_face = kwargs.get('restored_face')
                if self.face_restore_helper:
                    self.face_restore_helper.add_restored_face(restored_face)
                    self.result.set_result(
                        fuc_name='add_restored_face',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='add_restored_face',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'get_inverse_affine':
                if self.face_restore_helper:
                    self.face_restore_helper.get_inverse_affine()
                    self.result.set_result(
                        fuc_name='get_inverse_affine',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
                else:
                    self.result.set_result(
                        fuc_name='get_inverse_affine',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            elif name == 'paste_faces_to_input_image':
                upsample_img = kwargs.get('upsample_img', None)
                if self.face_restore_helper:
                    result_img = self.face_restore_helper.paste_faces_to_input_image(upsample_img)
                    self.result.set_result(
                        fuc_name='paste_faces_to_input_image',
                        is_success=True,
                        fail_reason='',
                        is_file=False,
                        file_path='',
                        except_data=result_img,
                        interface_return=result_img
                    )
                else:
                    self.result.set_result(
                        fuc_name='paste_faces_to_input_image',
                        is_success=False,
                        fail_reason='FaceRestoreHelper not initialized',
                        is_file=False,
                        file_path='',
                        except_data=None,
                        interface_return=None
                    )
            else:
                self.result.set_result(
                    fuc_name=name,
                    is_success=False,
                    fail_reason='Function not found',
                    is_file=False,
                    file_path='',
                    except_data=None,
                    interface_return=None
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

# 以下是自动生成的代码，请勿修改
# 为custom_adapter添加额外属性 additional_data
# 该属性用于存储函数名，等
adapter_additional_data = {}
adapter_additional_data['functions'] = []
adapter_additional_data['functions'].append('enhance')
adapter_additional_data['functions'].append('load_file_from_url')
adapter_additional_data['functions'].append('img2tensor')
adapter_additional_data['functions'].append('normalize')
adapter_additional_data['functions'].append('tensor2img')
adapter_additional_data['functions'].append('FaceRestoreHelper')
adapter_additional_data['functions'].append('clean_all')
adapter_additional_data['functions'].append('read_image')
adapter_additional_data['functions'].append('get_face_landmarks_5')
adapter_additional_data['functions'].append('align_warp_face')
adapter_additional_data['functions'].append('add_restored_face')
adapter_additional_data['functions'].append('get_inverse_affine')
adapter_additional_data['functions'].append('paste_faces_to_input_image')

original_init = CustomAdapter.__init__
def new_init(self):
    original_init(self)
    self.additional_data = adapter_additional_data
CustomAdapter.__init__ = new_init
if not os.path.exists(ENV_DIR):
    os.makedirs(ENV_DIR)
