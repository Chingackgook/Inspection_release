from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.EasyOCR import *
exe = Executor('EasyOCR', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/EasyOCR/easyocr/cli.py'
import argparse
import easyocr
import easyocr

def run_easyocr():
    lang_list = ['en']
    gpu = True
    model_storage_directory = None
    user_network_directory = None
    recog_network = 'standard'
    download_enabled = True
    detector = True
    recognizer = True
    verbose = True
    quantize = True
    input_file = RESOURCES_PATH + 'images/test_image.jpg'
    decoder = 'greedy'
    beamWidth = 5
    batch_size = 1
    workers = 0
    allowlist = None
    blocklist = None
    detail = 1
    rotation_info = None
    paragraph = False
    min_size = 20
    contrast_ths = 0.1
    adjust_contrast = 0.5
    text_threshold = 0.7
    low_text = 0.4
    link_threshold = 0.4
    canvas_size = 2560
    mag_ratio = 1.0
    slope_ths = 0.1
    ycenter_ths = 0.5
    height_ths = 0.5
    width_ths = 0.5
    y_ths = 0.5
    x_ths = 1.0
    add_margin = 0.1
    output_format = 'standard'
    reader = exe.create_interface_objects(interface_class_name='Reader', lang_list=lang_list, gpu=gpu, model_storage_directory=model_storage_directory, user_network_directory=user_network_directory, recog_network=recog_network, download_enabled=download_enabled, detector=detector, recognizer=recognizer, verbose=verbose, quantize=quantize)
    for line in exe.run('readtext', image=input_file, decoder=decoder, beamWidth=beamWidth, batch_size=batch_size, workers=workers, allowlist=allowlist, blocklist=blocklist, detail=detail, rotation_info=rotation_info, paragraph=paragraph, min_size=min_size, contrast_ths=contrast_ths, adjust_contrast=adjust_contrast, text_threshold=text_threshold, low_text=low_text, link_threshold=link_threshold, canvas_size=canvas_size, mag_ratio=mag_ratio, slope_ths=slope_ths, ycenter_ths=ycenter_ths, height_ths=height_ths, width_ths=width_ths, y_ths=y_ths, x_ths=x_ths, add_margin=add_margin, output_format=output_format):
        print(line)
run_easyocr()