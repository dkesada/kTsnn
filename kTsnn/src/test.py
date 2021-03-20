from kTsnn.src.utils import *
from kTsnn.src.nets.cnn import Conv1dnn
from kTsnn.src.nets.window_gen import WindowGenerator
import tensorflow as tf
import os
import random as rn
import numpy as np

MAX_EPOCHS = 50
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

    dt_train, dt_test, dt_val = get_train_test_val(dt, info['test'], info['val'], info['idx_cyc'])
    #dt_train, dt_test, dt_val = norm_dt(dt_train, dt_test, dt_val)

    num_features = dt_train.shape[1]

    # Forecasting into the future
    OUT_STEPS = 20
    INPUT_WIDTH = 7
    multi_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=OUT_STEPS, shift=OUT_STEPS,
                         dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                         label_columns=dt_train.columns)  #info['obj_var'])
    CONV_WIDTH = 7
    ini = 0

    model = Conv1dnn(MAX_EPOCHS, multi_window, num_features, conv_width=CONV_WIDTH, out_steps=OUT_STEPS)
    model.train_net()
    model.fit_net(patience=5)
    #pred = model.predict(dt_test.loc[ini:(ini+INPUT_WIDTH+OUT_STEPS), :], obj_var=info['obj_var'][0])
    path = model.predict_long_term(dt_test, obj_var=info['obj_var'][0], length=77)



