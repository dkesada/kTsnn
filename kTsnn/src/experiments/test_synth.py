from kTsnn.src.utils import *
import tensorflow as tf
import os
import random as rn
import numpy as np
from multiprocessing import Process, Queue
import multitasking
import signal

# Synthetic data experiments

DT_FILE = 'dt_synth_unfolded.csv'
INFO_FILE = 'exec_info_synth.txt'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(4242)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(424242)


if __name__ == '__main__':
    multitasking.set_engine("process")
    multitasking.set_max_threads(1)
    signal.signal(signal.SIGINT, multitasking.killall)

    dt = load_dt(DT_FILE)
    info = load_info(INFO_FILE)
    res = [[], [], [], []]

    # Settings
    # learning_rate=0.01 In the LSTM file
    out_steps = 100
    units = 128
    input_width = 1
    ini = 0
    length = 90
    max_epochs = 100
    patience = 10
    model_arch = None
    num_features = 1

    # model_arch = tf.keras.Sequential([
    #         tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #         tf.keras.layers.LSTM(units),
    #         tf.keras.layers.Dense(out_steps * num_features,
    #                               kernel_initializer=tf.initializers.zeros),
    #         tf.keras.layers.Reshape([out_steps, num_features])])

    # info["cv"] = info["cv"][1:2]
    # info["cv"][0]["test"] = [13]

    for cv in info['cv']:
        queue_cv = Queue()
        main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
                            out_steps, units, input_width, num_features, max_epochs, patience, model_arch,
                            mode=3, single=False, queue=queue_cv)
        cv_res = queue_cv.get()
        res[0].append(cv_res[0].mean())
        res[1].append(cv_res[1].mean())
        res[2].append(cv_res[2].mean())
        res[3].append(cv_res[3])

    print("Final MAE of the model: ")
    print(np.mean(res[0]))
    print("Final MAPE of the model: ")
    print(np.mean(res[1]))
    print("Final exec. time of the model: ")
    print(np.mean(res[2]))
    print("Final training time of the model: ")
    print(np.mean(res[3]))













