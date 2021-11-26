from kTsnn.src.utils import *
import tensorflow as tf
import os
import random as rn
import numpy as np

# Synthetic data experiments

DT_FILE = 'dt_synth_unfolded.csv'
INFO_FILE = 'exec_info_synth.txt'

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
    # learning_rate=0.01 In the LSTM file
    out_steps = 90
    units = 32
    input_width = 10
    ini = 0
    length = 90
    max_epochs = 300
    patience = 10
    model_arch = None
    num_features = dt.shape[1]-1

    model_arch = tf.keras.Sequential([
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(out_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([out_steps, num_features])])

    for cv in info['cv']:
        cv_res, _ = main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
                                        out_steps, units, input_width, max_epochs, patience, model_arch, mode=2)
        res.append(cv_res.mean())

    print(np.mean(res))













