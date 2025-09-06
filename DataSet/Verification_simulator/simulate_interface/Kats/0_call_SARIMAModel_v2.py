from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.Kats import *
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
from kats.consts import TimeSeriesData
from kats.models.sarima import SARIMAModel
from kats.models.sarima import SARIMAParams
exe = Executor('Kats', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = (
    '/mnt/autor_name/haoTingDeWenJianJia/Kats/tutorials/kats_201_forecasting.ipynb'
    )
warnings.simplefilter(action='ignore')
air_passengers_df = pd.read_csv('/mnt/autor_name/haoTingDeWenJianJia/Kats/kats/data/air_passengers.csv') # add 
air_passengers_df.columns = ['time', 'value']
air_passengers_ts = TimeSeriesData(air_passengers_df)
params = SARIMAParams(p=2, d=1, q=1, trend='ct', seasonal_order=(1, 0, 1, 12))
m = exe.create_interface_objects(interface_class_name='SARIMAModel', data=
    air_passengers_ts, params=params)
exe.run('fit')
fcst = exe.run('predict', steps=30, freq='MS', include_history=True)
m.plot()
