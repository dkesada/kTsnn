from kTsnn.src.utils import *
# import tensorflow as tf
import os
import random as rn
import numpy as np
from kTsnn.src.nets.lstm_rec import LSTM_rec
from multiprocessing import Process, Queue
import multitasking
import signal

# Real motor data experiments

DT_FILE = 'dt_motor_red_unfolded.csv'
INFO_FILE = 'exec_info_motor.txt'

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(4242)
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# tf.random.set_seed(424242)

if __name__ == '__main__':
    multitasking.set_engine("process")
    multitasking.set_max_threads(1)
    signal.signal(signal.SIGINT, multitasking.killall)

    dt = load_dt(DT_FILE)
    info = load_info(INFO_FILE)
    res = [[], [], []]

    # Settings
    out_steps = 1
    units = 128
    input_width = 2
    ini = 0
    length = 20
    max_epochs = 300
    patience = 10
    model_arch = None
    num_features = 11

    # model_arch = tf.keras.Sequential([
    #         #tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #         tf.keras.layers.LSTM(units, return_sequences=False),
    #         #tf.keras.layers.Dense(36,
    #         #                      kernel_initializer=tf.initializers.zeros),
    #         tf.keras.layers.Dense(out_steps * num_features,
    #                               kernel_initializer=tf.initializers.zeros),
    #         tf.keras.layers.Reshape([out_steps, num_features])])

    # model_arch = tf.keras.Sequential([
    #                 # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    #                 tf.keras.layers.Lambda(lambda x: x[:, -units:, :]),
    #                 # Shape => [batch, 1, conv_units]
    #                 tf.keras.layers.Conv1D(256, activation='relu', kernel_size=units),
    #                 # Shape => [batch, 1,  out_steps*features]
    #                 tf.keras.layers.Dense(out_steps * num_features,
    #                                       kernel_initializer=tf.initializers.zeros),
    #                 # Shape => [batch, out_steps, features]
    #                 tf.keras.layers.Reshape([out_steps, num_features])])

    # model_arch = LSTM_rec(units, out_steps, num_features)

    #info["cv"] = info["cv"][1:2]
    #info["cv"][0]["test"] = [13]

    for cv in info['cv']:
        queue_cv = Queue()
        main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
                            out_steps, units, input_width, num_features, max_epochs, patience, model_arch,
                            mode=2, single=False, queue=queue_cv)
        cv_res = queue_cv.get()
        res[0].append(cv_res[0].mean())
        res[1].append(cv_res[1].mean())
        res[2].append(cv_res[2])

    # for cv in info['cv']:
    #     cv_res, _ = main_pipeline_synth(dt, cv, info['idx_cyc'], info['obj_var'], ini, length,
    #                                     out_steps, units, input_width, num_features, max_epochs, patience, model_arch,
    #                                     mode=2, single=False)
    #     res[0].append(cv_res[0].mean())
    #     res[1].append(cv_res[1].mean())
    #     res[2].append(cv_res[2])

    print("Final MAE of the model: ")
    print(np.mean(res[0]))
    print("Final exec. time of the model: ")
    print(np.mean(res[1]))
    print("Final training time of the model: ")
    print(np.mean(res[2]))

