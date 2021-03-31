from kTsnn.src.utils import *
import tensorflow as tf
import os
import random as rn
import numpy as np


DT_FILE = 'dt_unfolded.csv'
INFO_FILE = 'exec_info.txt'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(4242)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(424242)


if __name__ == '__main__':
    dt = load_dt(DT_FILE)
    info = load_info(INFO_FILE)
    res = []

    # Settings
    out_steps = 24
    conv_width = 7
    input_width = 24
    ini = 0
    length = 50
    max_epochs = 300
    patience = 5
    model_arch = None

    # model_arch = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, lstm_units]
    #     # Adding more `lstm_units` just overfits more quickly.
    #     tf.keras.layers.LSTM(32, return_sequences=False),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])

    for cv in info['cv']:
        cv_res, _ = main_pipeline(dt, cv, info['idx_cyc'], info['obj_var'][0],
                                  ini, length, out_steps, conv_width, input_width, max_epochs, patience, model_arch)
        res.append(cv_res.mean())

    print(np.mean(res))













