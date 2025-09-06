from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.rembg import *
exe = Executor('rembg', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/rembg/rembg/commands/i_command.py'
# Import the existing package
import json
import sys
from typing import IO
import click
from rembg.bg import remove
from rembg.session_factory import new_session
from rembg.sessions import sessions_names
# end

def i_command(model: str='u2net', extras: str='', input_path: str='input.jpg', output_path: str=FILE_RECORD_PATH + '/output.png', **kwargs) -> None:
    """
    Function to process an input file based on the provided options.

    This function reads an input file, applies image processing operations based on the provided options, and writes the output to a file.

    Parameters:
        model (str): The name of the model to use for image processing.
        extras (str): Additional options in JSON format.
        input_path (str): The input file path to process.
        output_path (str): The output file path to write the processed image to.
        **kwargs: Additional keyword arguments corresponding to the command line options.

    Returns:
        None
    """
    kwargs = {'alpha_matting': True, 'alpha_matting_foreground_threshold': 240, 'alpha_matting_background_threshold': 10, 'alpha_matting_erode_size': 10, 'only_mask': False, 'post_process_mask': False, 'bgcolor': (0, 0, 0, 0), **kwargs}
    try:
        kwargs.update(json.loads(extras))
    except Exception:
        pass
    with open(input_path, 'rb') as input_file, open(output_path, 'wb') as output_file:
        output_data = exe.run('remove', data=input_file.read(), session=new_session(model, **kwargs), **kwargs)
        output_file.write(output_data)

# Run the main logic directly
i_command(input_path='input.jpg', output_path=FILE_RECORD_PATH + '/output.png')
