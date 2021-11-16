from kTsnn.src.utils import *
import tensorflow as tf
import os
import random as rn
import numpy as np

# Synthetic data experiments

DT_FILE = 'dt_synth_unfolded.csv'
INFO_FILE = 'exec_info_synth.txt'
VAL_PER = 0.1

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
    out_steps = 90
    conv_width = 6
    input_width = 10
    ini = 0
    length = 90
    max_epochs = 300
    patience = 10
    model_arch = None
    num_features = dt.shape[1]-1

    # model_arch = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, lstm_units]
    #     # Adding more `lstm_units` just overfits more quickly.
    #     tf.keras.layers.LSTM(16, return_sequences=False),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(out_steps , kernel_initializer=tf.initializers.zeros()),
    #     tf.keras.layers.Dense(out_steps * num_features,
    #                           kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([out_steps, num_features])
    # ])

    for cv in info['cv']:
        cv_res, _ = main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
                                        out_steps, conv_width, input_width, max_epochs, patience, model_arch)
        res.append(cv_res.mean())

    print(np.mean(res))













