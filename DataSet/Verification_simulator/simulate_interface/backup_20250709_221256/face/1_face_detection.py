
import numpy as np
import sys
import os
from Inspection.core.executor import Executor
from Inspection.adapters.custom_adapters.face import ENV_DIR
exe = Executor('face', 'simulation')
FILE_RECORD_PATH = exe.now_record_path

# -*- coding: utf-8 -*-
import click
import os
import re
import multiprocessing
import itertools
import time


def print_result(filename, location):
    top, right, bottom, left = location
    print("{},{},{},{},{}".format(filename, top, right, bottom, left))


know_image = exe.run("load_image_file", file=ENV_DIR + "masike.jpg")
known_face_encodings = exe.run("face_encodings", face_image=know_image)  # 确保face_encodings是已实现的方法

def test_image(image_to_check, model, upsample):
    unknown_image = exe.run("load_image_file", file=image_to_check)  # 确保load_image_file是已实现的方法
    face_locations = exe.run("face_locations", img=unknown_image, number_of_times_to_upsample=upsample, model=model)  # 确保face_locations是已实现的方法

    for face_location in face_locations:
        print_result(image_to_check, face_location)
    face_encodings = exe.run("face_encodings", face_image=unknown_image, known_face_locations=face_locations)  # 确保face_encodings是已实现的方法
    # 这里需要确保known_face_encodings是已定义的变量，可能需要在函数中定义或传入
    if len(face_encodings) == 0:
        print("WARNING: No faces found in {}. Ignoring file.".format(image_to_check))
        return
    results = exe.run("compare_faces", known_face_encodings=known_face_encodings, face_encoding_to_check=face_encodings[0])  # 确保compare_faces是已实现的方法
    # 这里可以根据需要使用结果，比如打印
    print(results)


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def process_images_in_process_pool(images_to_check, number_of_cpus, model, upsample):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(model),
        itertools.repeat(upsample),
    )

    pool.starmap(test_image, function_parameters)


def main(images_floder, cpus=1, model="hog", upsample=0):
    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1

    if os.path.isdir(images_floder):
        if cpus == 1:
            [test_image(image_file, model, upsample) for image_file in image_files_in_folder(images_floder)]
        else:
            process_images_in_process_pool(image_files_in_folder(images_floder), cpus, model, upsample)
    else:
        test_image(images_floder, model, upsample)


# 直接运行主逻辑
images_floder = ENV_DIR  # 这是一个文件夹，内有多张图片，不能直接作为load_image_file的参数
main(images_floder)
