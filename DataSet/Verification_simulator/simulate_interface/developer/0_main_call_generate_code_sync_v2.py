from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.developer import *
exe = Executor('developer', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/developer/smol_dev/main.py'
import sys
import time
from smol_dev.prompts import plan
from smol_dev.prompts import specify_file_paths
from smol_dev.prompts import generate_code_sync
from smol_dev.utils import generate_folder
from smol_dev.utils import write_file
import argparse
import sys
import time
from smol_dev.utils import generate_folder, write_file
defaultmodel = 'gpt-4o-mini'

def main(prompt, generate_folder_path='generated', debug=False, model: str=defaultmodel):
    generate_folder(generate_folder_path)
    if debug:
        print('--------shared_deps---------')
    shared_deps_file_path = f'{FILE_RECORD_PATH}/shared_deps.md'
    with open(shared_deps_file_path, 'wb') as f:
        start_time = time.time()

        def stream_handler(chunk):
            f.write(chunk)
            if debug:
                end_time = time.time()
                sys.stdout.write('\r \x1b[93mChars streamed\x1b[0m: {}. \x1b[93mChars per second\x1b[0m: {:.2f}'.format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                sys.stdout.flush()
                stream_handler.count += len(chunk)
        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write('\x1b[0m\n')
        shared_deps = exe.run('plan', prompt=prompt, stream_handler=stream_handler, model=model)
    if debug:
        print(shared_deps)
    write_file(shared_deps_file_path, shared_deps)
    if debug:
        print('--------shared_deps---------')
        print('--------specify_filePaths---------')
    file_paths = exe.run('specify_file_paths', prompt=prompt, plan=shared_deps, model=model)
    if debug:
        print(file_paths)
        print('--------file_paths---------')
    for file_path in file_paths:
        file_path = f'{FILE_RECORD_PATH}/{file_path}'
        if debug:
            print(f'--------generate_code: {file_path} ---------')
        start_time = time.time()

        def stream_handler(chunk):
            if debug:
                end_time = time.time()
                sys.stdout.write('\r \x1b[93mChars streamed\x1b[0m: {}. \x1b[93mChars per second\x1b[0m: {:.2f}'.format(stream_handler.count, stream_handler.count / (end_time - start_time)))
                sys.stdout.flush()
                stream_handler.count += len(chunk)
        stream_handler.count = 0
        stream_handler.onComplete = lambda x: sys.stdout.write('\x1b[0m\n')
        code = exe.run('generate_code_sync', prompt=prompt, plan=shared_deps, current_file=file_path, stream_handler=stream_handler, model=model)
        if debug:
            print(code)
            print(f'--------generate_code: {file_path} ---------')
        write_file(file_path, code)
    print('--------smol dev done!---------')
prompt = "\n  a simple JavaScript/HTML/CSS/Canvas app that is a one player game of PONG. \n  The left paddle is controlled by the player, following where the mouse goes.\n  The right paddle is controlled by a simple AI algorithm, which slowly moves the paddle toward the ball at every frame, with some probability of error.\n  Make the canvas a 400 x 400 black square and center it in the app.\n  Make the paddles 100px long, yellow and the ball small and red.\n  Make sure to render the paddles and name them so they can controlled in javascript.\n  Implement the collision detection and scoring as well.\n  Every time the ball bouncess off a paddle, the ball should move faster.\n  It is meant to run in Chrome browser, so dont use anything that is not supported by Chrome, and don't use the import and export keywords.\n"
generate_folder_path = 'generated'
debug = True
main(prompt=prompt, generate_folder_path=generate_folder_path, debug=debug)