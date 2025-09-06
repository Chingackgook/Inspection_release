from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Surprise import *
import sys
exe = Executor('Surprise', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/Surprise/examples/predict_ratings.py'
from surprise import Dataset
from surprise import KNNBasic
'\nThis module describes how to train on a full dataset (when no testset is\nbuilt/specified) and how to use the predict() method.\n'

def main():
    data = Dataset.load_builtin('ml-100k')
    trainset = data.build_full_trainset()
    algo = exe.create_interface_objects(interface_class_name='KNNBasic', k=40, min_k=1, sim_options={}, verbose=True)
    exe.run('fit', trainset=trainset)
    uid = str(196)
    iid = str(302)
    pred = exe.run('predict', uid=uid, iid=iid, r_ui=4, verbose=True)
main()