import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.EasyOCR import ENV_DIR
from Inspection.adapters.custom_adapters.EasyOCR import *
exe = Executor('EasyOCR', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
import easyocr
import cv2

# 初始化Reader并加载模型
# 这里调用exe.create_interface_objects来加载模型
exe.create_interface_objects(lang_list=['ch_sim', 'en'])

# 读取图像
image_path = os.path.join(ENV_DIR, 'examples/chinese.jpg')
img = cv2.imread(image_path)
img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用exe.run替换函数调用
detected_texts = exe.run("detect", img=img, min_size=20, text_threshold=0.7, low_text=0.4, link_threshold=0.4, canvas_size=2560)
recognized_texts = exe.run("recognize", img_cv_grey=img_cv_grey, horizontal_list=None, free_list=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1)
result = exe.run("readtext", image=image_path, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1)
result_lang = exe.run("readtextlang", image=image_path, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1)
result_batched = exe.run("readtext_batched", image=image_path, n_width=None, n_height=None, decoder='greedy', beamWidth=5, batch_size=1, workers=0, allowlist=None, blocklist=None, detail=1)

# 获取检测器路径
detector_path = exe.run("getDetectorPath", detect_network="craft")

# 设置检测器
exe.run("setDetector", detect_network="craft")

# 设置模型语言
exe.run("setModelLanguage", language='ch_sim', lang_list=['ch_sim', 'en'], list_lang=None, list_lang_string=None)

# 获取字符
char = exe.run("getChar", fileName=os.path.join(ENV_DIR, 'path/to/char_file.txt'))

# 设置语言列表
exe.run("setLanguageList", lang_list=['ch_sim', 'en'], model=None)

# 打印结果
print("Detected Texts:", detected_texts)
print("Recognized Texts:", recognized_texts)
print("Read Text Result:", result)
print("Read Text Language Result:", result_lang)
print("Read Text Batched Result:", result_batched)
print("Detector Path:", detector_path)
