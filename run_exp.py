import sys
import os
# builtin
import importlib
from datetime import datetime
# internal
import read_data
import train_model
import params
from params import DATASET, MODEL, LR, LAMDA, LAYER, BATCH_SIZE
from tqdm.contrib.concurrent import process_map
# external
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.compat.v1.disable_eager_execution() # to disable the eager mode

work_space_dir = './'
res_dir = work_space_dir + f"experiment_result/{DATASET}/"
target_dir = res_dir + f'{MODEL}/'
# check
# assert params.all_para[2] == 'LightRGCN'
print(params.all_para)
# read
read_data_res_tri = read_data.read_all_data_tri(params.all_para, approximate=False)
# run
today = datetime.today()
formatted_date = today.strftime('%Y%m%d')
for i in range(3):
    exsl_path = f'{DATASET}_{MODEL}_{formatted_date}_{LR}_{LAMDA}_{LAYER}_{BATCH_SIZE}_{i}.xlsx'
    F1_max = train_model.train_model(params.all_para[:26], read_data_res_tri, target_dir + exsl_path, '')
