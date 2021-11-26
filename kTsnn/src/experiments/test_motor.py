from kTsnn.src.utils import *
import tensorflow as tf
import os
import random as rn
import numpy as np

# Synthetic data experiments

DT_FILE = 'dt_motor_red_unfolded.csv'
INFO_FILE = 'exec_info_motor.txt'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(4242)
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.random.set_seed(424242)


if __name__ == '__main__':
    dt = load_dt(DT_FILE)
    info = load_info(INFO_FILE)
    res = [[], [], []]

    # Settings
    out_steps = 20
    units = 32
    input_width = 3
    ini = 0
    length = 20
    max_epochs = 300
    patience = 10
    model_arch = None
    num_features = dt.shape[1]-1

    model_arch = tf.keras.Sequential([
            #tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dense(out_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([out_steps, num_features])])

    for cv in info['cv']:
        cv_res, _ = main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
                                        out_steps, units, input_width, max_epochs, patience, model_arch, mode=1)
        res[0].append(cv_res[0].mean())
        res[1].append(cv_res[1].mean())
        res[2].append(cv_res[2])

    print("Final MAE of the model: ")
    print(np.mean(res[0]))
    print("Final exec. time of the model: ")
    print(np.mean(res[1]))
    print("Final training time of the model: ")
    print(np.mean(res[2]))














