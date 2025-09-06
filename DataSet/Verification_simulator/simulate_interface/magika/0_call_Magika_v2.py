from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.magika import *
exe = Executor('magika', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/magika/python/src/magika/cli/magika_client.py'
import importlib.metadata
import json
import logging
import os
import sys
from pathlib import Path
from typing import List
from typing import Optional
from typing import Tuple
import click
from magika import Magika
from magika import MagikaError
from magika import PredictionMode
from magika import colors
from magika.logger import get_logger
from magika.types import ContentTypeLabel
from magika.types import MagikaResult
from magika.types.overwrite_reason import OverwriteReason
VERSION = importlib.metadata.version('magika')

def run_magika_analysis():
    files_paths = [Path('')]
    recursive = False
    json_output = False
    jsonl_output = False
    mime_output = False
    label_output = False
    magic_compatibility_mode = False
    output_score = False
    prediction_mode_str = PredictionMode.HIGH_CONFIDENCE
    batch_size = 32
    no_dereference = False
    with_colors = True
    verbose = False
    debug = False
    output_version = False
    model_dir = None
    if magic_compatibility_mode:
        with_colors = False
    _l = get_logger(use_colors=with_colors)
    if verbose:
        _l.setLevel(logging.INFO)
    if debug:
        _l.setLevel(logging.DEBUG)
    if output_version:
        _l.raw_print_to_stdout('Magika python client')
        _l.raw_print_to_stdout(f'Magika version: {VERSION}')
        _l.raw_print_to_stdout(f'Default model: {exe.run('get_model_name')}')
        return
    if len(files_paths) == 0:
        _l.error('You need to pass at least one path, or - to read from stdin.')
        return
    read_from_stdin = False
    for p in files_paths:
        if str(p) == '-':
            read_from_stdin = True
        elif not p.exists():
            _l.error(f'File or directory "{str(p)}" does not exist.')
            return
    if read_from_stdin:
        if len(files_paths) > 1:
            _l.error('If you pass "-", you cannot pass anything else.')
            return
        if recursive:
            _l.error('If you pass "-", recursive scan is not meaningful.')
            return
    if batch_size <= 0 or batch_size > 512:
        _l.error('Batch size needs to be greater than 0 and less or equal than 512.')
        return
    if json_output and jsonl_output:
        _l.error('You should use either --json or --jsonl, not both.')
        return
    if int(mime_output) + int(label_output) + int(magic_compatibility_mode) > 1:
        _l.error('You should use only one of --mime, --label, --compatibility-mode.')
        return
    if recursive:
        expanded_paths = []
        for p in files_paths:
            if p.exists():
                if p.is_file():
                    expanded_paths.append(p)
                elif p.is_dir():
                    expanded_paths.extend(sorted(p.rglob('*')))
            elif str(p) == '-':
                pass
            else:
                _l.error(f'File or directory "{str(p)}" does not exist.')
                return
        files_paths = list(filter(lambda x: not x.is_dir(), expanded_paths))
    _l.info(f'Considering {len(files_paths)} files')
    _l.debug(f'Files: {files_paths}')
    if model_dir is None:
        model_dir_str = os.environ.get('MAGIKA_MODEL_DIR')
        if model_dir_str is not None and model_dir_str.strip() != '':
            model_dir = Path(model_dir_str)
    try:
        magika = exe.create_interface_objects(interface_class_name='Magika', model_dir=model_dir, prediction_mode=PredictionMode(prediction_mode_str), no_dereference=no_dereference, verbose=verbose, debug=debug, use_colors=with_colors)
    except MagikaError as mr:
        _l.error(str(mr))
        return
    start_color = ''
    end_color = ''
    color_by_group = {'document': colors.LIGHT_PURPLE, 'executable': colors.LIGHT_GREEN, 'archive': colors.LIGHT_RED, 'audio': colors.YELLOW, 'image': colors.YELLOW, 'video': colors.YELLOW, 'code': colors.LIGHT_BLUE}
    all_predictions: List[Tuple[Path, MagikaResult]] = []
    batches_num = len(files_paths) // batch_size
    if len(files_paths) % batch_size != 0:
        batches_num += 1
    for batch_idx in range(batches_num):
        batch_files_paths = files_paths[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        if should_read_from_stdin(files_paths):
            batch_predictions = [get_magika_result_from_stdin(magika)]
        else:
            batch_predictions = exe.run('identify_paths', paths=batch_files_paths)
        if json_output:
            all_predictions.extend(zip(batch_files_paths, batch_predictions))
        elif jsonl_output:
            for file_path, result in zip(batch_files_paths, batch_predictions):
                _l.raw_print_to_stdout(json.dumps(result.asdict()))
        else:
            for file_path, result in zip(batch_files_paths, batch_predictions):
                if result.ok:
                    if mime_output:
                        output = result.prediction.output.mime_type
                    elif label_output:
                        output = str(result.prediction.output.label)
                    else:
                        output = f'{result.prediction.output.description} ({result.prediction.output.group})'
                        if result.prediction.dl.label != ContentTypeLabel.UNDEFINED and result.prediction.dl.label != result.prediction.output.label and (result.prediction.overwrite_reason == OverwriteReason.LOW_CONFIDENCE):
                            output += f' [Low-confidence model best-guess: {result.prediction.dl.description} ({result.prediction.dl.group}), score={result.prediction.score}]'
                    if with_colors:
                        start_color = color_by_group.get(result.prediction.output.group, colors.WHITE)
                        end_color = colors.RESET
                else:
                    output = result.status
                    start_color = ''
                    end_color = ''
                if output_score and result.ok:
                    score = int(result.prediction.score * 100)
                    _l.raw_print_to_stdout(f'{start_color}{FILE_RECORD_PATH}/{file_path.name}: {output} {score}%{end_color}')
                else:
                    _l.raw_print_to_stdout(f'{start_color}{FILE_RECORD_PATH}/{file_path.name}: {output}{end_color}')
    if json_output:
        _l.raw_print_to_stdout(json.dumps([result.asdict() for _, result in all_predictions], indent=4))

def should_read_from_stdin(files_paths: List[Path]) -> bool:
    return len(files_paths) == 1 and str(files_paths[0]) == '-'

def get_magika_result_from_stdin(magika: Magika) -> MagikaResult:
    content = sys.stdin.buffer.read()
    result = exe.run('identify_bytes', content=content)
    return result
run_magika_analysis()