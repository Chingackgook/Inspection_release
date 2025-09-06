from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.autokeras import *
import keras
import numpy as np
import autokeras as ak
exe = Executor('autokeras', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/autokeras/examples/imdb.py'

def imdb_raw():
    max_features = 20000
    index_offset = 3
    (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features, index_from=index_offset)
    x_train = x_train[:2]
    y_train = y_train.reshape(-1, 1)[:2]
    x_test = x_test[:1]
    y_test = y_test.reshape(-1, 1)[:1]
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k: v + index_offset for k, v in word_to_id.items()}
    word_to_id['<PAD>'] = 0
    word_to_id['<START>'] = 1
    word_to_id['<UNK>'] = 2
    id_to_word = {value: key for key, value in word_to_id.items()}
    x_train = list(map(lambda sentence: ' '.join((id_to_word[i] for i in sentence)), x_train))
    x_test = list(map(lambda sentence: ' '.join((id_to_word[i] for i in sentence)), x_test))
    x_train = np.array(x_train, dtype=str)
    x_test = np.array(x_test, dtype=str)
    return ((x_train, y_train), (x_test, y_test))
(x_train, y_train), (x_test, y_test) = imdb_raw()
print(x_train.shape)
print(y_train.shape)
print(x_train[0][:50])
clf = exe.create_interface_objects(interface_class_name='TextClassifier', max_trials=3)
history = exe.run('fit', x=x_train, y=y_train, epochs=1, batch_size=1)
accuracy = exe.run('evaluate', x=x_test, y=y_test)
print('Accuracy: {accuracy}'.format(accuracy=accuracy))