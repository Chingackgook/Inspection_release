from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.pyvideotrans import *
exe = Executor('pyvideotrans', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/pyvideotrans/cli.py'
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs
from urllib.parse import urlparse
import zhconv
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import zhconv

def download_file(url):
    """Downloads a file from a URL and saves it to FILE_RECORD_PATH."""
    if sys.platform != 'linux':
        raise Exception('仅在 Google Colab 上可下载文件，其他系统请传递文件绝对路径')
    parsed_url = urlparse(url)
    filename = None
    filepath = None
    Path(FILE_RECORD_PATH).mkdir(exist_ok=True)
    if parsed_url.path:
        potential_filename = os.path.basename(parsed_url.path)
        if '.' in potential_filename:
            filename = re.sub('[^\\w\\-_\\.]', '', potential_filename)
            filepath = os.path.join(FILE_RECORD_PATH, filename)
    if not filepath:
        query_params = parse_qs(parsed_url.query)
        video_audio_exts = ['mp4', 'mov', 'mkv', 'mpeg', 'avi', 'wmv', 'ts', 'wav', 'flac', 'mp3', 'm4a', 'wma']
        for param_value in query_params.values():
            for value in param_value:
                potential_filename_with_ext = None
                for ext in video_audio_exts:
                    if '.' + ext in value:
                        potential_filename_with_ext = value
                        break
                if potential_filename_with_ext:
                    filename = re.sub('[^\\w\\-_\\.]', '', potential_filename_with_ext)
                    filepath = os.path.join(FILE_RECORD_PATH, filename)
                    break
    if filepath and filename:
        try:
            subprocess.run(['wget', '-O', filepath, url], check=True, capture_output=True)
            return filepath
        except subprocess.CalledProcessError as e:
            print(f'Error downloading file: {e.stderr.decode()}')
            return None
    else:
        print('No valid filename found in URL.')
        return None
model_name = 'tiny'
language = 'auto'
audio_file = RESOURCES_PATH + 'audios/test_audio.wav'
device = 'auto'
compute_type = 'default'
prompt = None
try:
    exe.run('speech_to_text', model_name=model_name, language=language, prompt=prompt, audio_file=audio_file, device=device, compute_type=compute_type)
except Exception as e:
    print(f'An error occurred: {e}')