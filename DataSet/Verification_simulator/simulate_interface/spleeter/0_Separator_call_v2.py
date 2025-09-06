from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.spleeter import *
exe = Executor('spleeter','simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/spleeter/tests/test_separator.py'
# Import the existing package
import itertools
from os.path import basename
from os.path import exists
from os.path import join
from os.path import splitext
from tempfile import TemporaryDirectory
import numpy as np
import pytest
import tensorflow as tf
from spleeter import SpleeterError
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
# end


""" Unit testing for Separator class. """
__email__ = 'spleeter@deezer.com'
__author__ = 'Deezer Research'
__license__ = 'MIT License'
import itertools
from os.path import basename, exists, join, splitext
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow as tf
from spleeter import SpleeterError
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
TEST_AUDIO_DESCRIPTORS = ['audio_example.mp3', 'audio_example_mono.mp3']
MODELS = ['spleeter:2stems', 'spleeter:4stems', 'spleeter:5stems']
MODEL_TO_INST = {'spleeter:2stems': ('vocals', 'accompaniment'),
    'spleeter:4stems': ('vocals', 'drums', 'bass', 'other'),
    'spleeter:5stems': ('vocals', 'drums', 'bass', 'piano', 'other')}
MODELS_AND_TEST_FILES = list(itertools.product(TEST_AUDIO_DESCRIPTORS, MODELS))
TEST_CONFIGURATIONS = list(itertools.product(TEST_AUDIO_DESCRIPTORS, MODELS))
print('RUNNING TESTS WITH TF VERSION {}'.format(tf.__version__))


def run_tests():
    for test_file, configuration in TEST_CONFIGURATIONS:
        test_separate(test_file, configuration)
        test_separate_to_file(test_file, configuration)
        test_filename_format(test_file, configuration)
    for test_file, configuration in MODELS_AND_TEST_FILES:
        test_filename_conflict(test_file, configuration)


def test_separate(test_file, configuration):
    """Test separation from raw data."""
    instruments = MODEL_TO_INST[configuration]
    adapter = AudioAdapter.default()
    waveform, _ = adapter.load(test_file)
    separator = exe.create_interface_objects(interface_class_name=
        'Separator', params_descriptor=configuration, MWF=False,
        multiprocess=False)
    prediction = exe.run('separate', waveform=waveform, audio_descriptor=
        test_file)
    assert len(prediction) == len(instruments)
    for instrument in instruments:
        assert instrument in prediction
    for instrument in instruments:
        track = prediction[instrument]
        assert waveform.shape[:-1] == track.shape[:-1]
        assert not np.allclose(waveform, track)
        for compared in instruments:
            if instrument != compared:
                assert not np.allclose(track, prediction[compared])


def test_separate_to_file(test_file, configuration):
    """Test file based separation."""
    instruments = MODEL_TO_INST[configuration]
    separator = exe.create_interface_objects(interface_class_name=
        'Separator', params_descriptor=configuration, MWF=False,
        multiprocess=False)
    name = splitext(basename(test_file))[0]
    with TemporaryDirectory() as directory:
        # Replace output path with FILE_RECORD_PATH
        exe.run('separate_to_file', audio_descriptor=test_file, destination=
            FILE_RECORD_PATH)
        for instrument in instruments:
            assert exists(join(FILE_RECORD_PATH, '{}/{}.wav'.format(name, instrument))
                )


def test_filename_format(test_file, configuration):
    """Test custom filename format."""
    instruments = MODEL_TO_INST[configuration]
    separator = exe.create_interface_objects(interface_class_name=
        'Separator', params_descriptor=configuration, MWF=False,
        multiprocess=False)
    name = splitext(basename(test_file))[0]
    with TemporaryDirectory() as directory:
        # Replace output path with FILE_RECORD_PATH
        exe.run('separate_to_file', audio_descriptor=test_file, destination=
            FILE_RECORD_PATH, filename_format=
            'export/{filename}/{instrument}.{codec}')
        for instrument in instruments:
            assert exists(join(FILE_RECORD_PATH, 'export/{}/{}.wav'.format(name,
                instrument)))


def test_filename_conflict(test_file, configuration):
    """Test error handling with static pattern."""
    separator = exe.create_interface_objects(interface_class_name=
        'Separator', params_descriptor=configuration, MWF=False,
        multiprocess=False)
    with TemporaryDirectory() as directory:
        with pytest.raises(SpleeterError):
            exe.run('separate_to_file', audio_descriptor=test_file,
                destination=directory, filename_format='I wanna be your lover')


# Directly run the main logic
run_tests()
