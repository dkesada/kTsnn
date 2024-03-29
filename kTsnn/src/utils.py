import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import numpy as np
import time
from kTsnn.src.nets.cnn import Conv1dnn
from kTsnn.src.nets.lstm import AutoLSTM
from kTsnn.src.nets.window_gen import WindowGenerator
from kTsnn.src.nets.lstm_rec import LSTM_rec
import multitasking

# Line plot of a column in a dataframe
def plot_col(dt, col):
    plt.figure()
    plt.plot(dt[col])
    plt.xlabel('Time')
    plt.ylabel(col)
    plt.show()


# Map function that returns a list with the result
def map_w(func, col):
    return list(map(func, col))


# Return the x and y dataframes for both train and validation
# The dataframe is divided by cycles, and each cycle is part of
# a crossvalidation fold
def get_train_test_val(dt, test, val, cyc_col):
    dt_test = dt[map_w(lambda x: x in test, dt[cyc_col])]
    dt_val = dt[map_w(lambda x: x in val, dt[cyc_col])]
    dt_train = dt[map_w(lambda x: x not in (test + val), dt[cyc_col])]
    cyc_idx_test = dt_test[cyc_col]
    dt_train = dt_train.drop(columns=cyc_col)
    dt_test = dt_test.drop(columns=cyc_col)
    dt_val = dt_val.drop(columns=cyc_col)

    return dt_train, dt_test, dt_val, cyc_idx_test


# Old one
def get_train_test(dt, cv, obj_col, cyc_col):
    in_test = map_w(lambda x: x in cv, dt[cyc_col])
    dt_test = dt[in_test]
    y_test = dt_test[obj_col]
    dt_train = dt[map_w(lambda x: not x, in_test)]
    y_train = dt_train[obj_col]
    dt_test = dt_test.drop(columns=[cyc_col] + obj_col)
    dt_train = dt_train.drop(columns=[cyc_col] + obj_col)

    return dt_test, y_test, dt_train, y_train


# Find the columns that have the given string in their names
def grep_columns(dt, sub):
    return dt.columns[map_w(lambda x: sub in x, dt.columns)]


# Load a dataset stored in the 'data' folder
def load_dt(file):
    return pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/" + file))


# Load the json info file. Its structure is {'obj_var': [...], 'idx_cyc': ..., 'cv': [[...],[...],...]}
def load_info(file):
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/" + file)) as f:
        info = json.load(f)
    return info


# Normalizes the train, test and validation datasets only using the mean and std of the train set
def norm_dt(dt_train, dt_test, dt_val, obj_var):
    dt_mean = dt_train.mean()
    dt_sd = dt_train.std()
    dt_train = (dt_train - dt_mean) / dt_sd
    dt_test = (dt_test - dt_mean) / dt_sd
    dt_val = (dt_val - dt_mean) / dt_sd
    dt_mean = dt_mean[dt_train.columns.get_loc(obj_var)]
    dt_sd = dt_sd[dt_train.columns.get_loc(obj_var)]

    return dt_train, dt_test, dt_val, dt_mean, dt_sd


def norm_dt_min_max(dt_train, dt_test, dt_val, obj_var):
    dt_min = dt_train.min()
    dt_max = dt_train.max()
    dt_train = (dt_train - dt_min) / (dt_max - dt_min)
    dt_test = (dt_test - dt_min) / (dt_max - dt_min)
    dt_val = (dt_val - dt_min) / (dt_max - dt_min)
    dt_min = dt_min[dt_train.columns.get_loc(obj_var)]
    dt_max = dt_max[dt_train.columns.get_loc(obj_var)]

    return dt_train, dt_test, dt_val, dt_min, dt_max


def mae(orig, pred):
    tmp = pred - orig
    tmp = np.abs(tmp)
    return tmp.sum() / len(tmp)


def mape(orig, pred):
    tmp = (orig - pred) / orig
    tmp = np.abs(tmp)
    return (100 / len(tmp)) * tmp.sum()


def undo_norm(ts, mean, sd):
    return ts * sd + mean


def undo_norm_min_max(ts, min, max):
    return ts * (max - min) + min


def eval_pred(orig, pred, mean, sd):
    if(len(orig) > len(pred)):
        orig = orig[0:len(pred)]
    # orig_un = undo_norm(orig, mean, sd)
    # pred_un = undo_norm(pred, mean, sd)
    orig_un = undo_norm_min_max(orig, mean, sd)
    pred_un = undo_norm_min_max(pred, mean, sd)
    res_mae = mae(orig_un, pred_un)
    res_mape = mape(orig_un, pred_un)
    print("MAE: {:f}".format(res_mae))
    print("MAPE: {:f}".format(res_mape))

    return res_mae, res_mape


def eval_model_single(model, dt_test, ini, length, obj_var, mean, sd, show_plot=False):
    orig = dt_test[obj_var]
    path = model.predict_single_shot(dt_test.iloc[ini:(ini+length+model.get_input_width()), :],
                                   obj_var=obj_var, length=length, show_plot=show_plot)
    res_mae, res_mape = eval_pred(orig, path, mean, sd)

    return res_mae, res_mape


def eval_model(model, dt_test, ini, length, obj_var, mean, sd, show_plot=False):
    obj_idx = 0
    if model.get_num_labels() != 1:
        obj_idx = dt_test.columns.get_loc(obj_var)
    orig = dt_test[obj_var][ini:((ini+length+model.get_input_width()))]
    path = model.predict_long_term(dt_test.iloc[ini:((ini+length+model.get_input_width())), :],
                                   obj_var=obj_var, length=length, show_plot=show_plot)
    res_mae, res_mape = eval_pred(orig, path[0:len(orig), obj_idx], mean, sd)

    return res_mae, res_mape


def eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, mean, sd, show_plot=False):
    res = np.zeros(len(cyc_idx_test.unique()))
    j = 0
    for i in cyc_idx_test.unique():
        rows = cyc_idx_test == i
        tmp = time.time()
        res[j] = eval_model(model, dt_test[rows], ini, length, obj_var, mean, sd, show_plot)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))
        j += 1

    return res


# Predicts from the initial point to the end of the dataset, taking intervals of 'length'+input size
def predict_long_term_full(self, dt, ini, obj_var, length, show_plot=True):
    n_preds = (dt.shape[0] / (self.get_input_width() + length)) // 1
    res = np.array([np.zeros(n_preds), np.zeros(n_preds)])
    for i in range(n_preds):
        tmp = time.time()
        path = self.predict_long_term(dt.iloc[ini:(len(dt)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    return res


# In case we do more than one forecasting per cycle
def eval_test_rep(model, dt_test, cyc_idx_test, ini, length, obj_var, mean, sd, show_plot=False, single=False):
    cycles = len(cyc_idx_test.unique())
    res = np.array([np.zeros(cycles), np.zeros(cycles), np.zeros(cycles)]) # MAE, MAPE, exec_time
    j = 0
    pad_size = model.get_input_width() + length
    for i in cyc_idx_test.unique():
        rows = cyc_idx_test == i
        n_preds = (dt_test[rows].shape[0] - ini) // pad_size
        cyc_res = np.array([np.zeros(n_preds), np.zeros(n_preds), np.zeros(n_preds)]) # MAE, MAPE, exec_time
        for k in range(n_preds):
            tmp = time.time()
            if single:
                cyc_res[0][k], cyc_res[1][k] = eval_model_single(model, dt_test[rows], ini + k * pad_size,
                                                                 length, obj_var, mean, sd, show_plot)
            else:
                cyc_res[0][k], cyc_res[1][k] = eval_model(model, dt_test[rows], ini + k * pad_size,
                                                          length, obj_var, mean, sd, show_plot)
            cyc_res[2][k] = time.time() - tmp
            print("Elapsed forecasting time: {:f} seconds".format(cyc_res[1][k]))

        res[0][j] = cyc_res[0].mean()
        res[1][j] = cyc_res[1].mean()
        res[2][j] = cyc_res[2].mean()
        j += 1

    return res


def main_pipeline(dt, cv, idx_cyc, obj_var, ini, length, out_steps, conv_width, input_width,
                  max_epochs, patience, model_arch=None, mode=3):
    # Obtain the correspondent cycles from the dataset
    dt_train, dt_test, dt_val, cyc_idx_test = get_train_test_val(dt, cv['test'], cv['val'], idx_cyc)
    dt_train, dt_test, dt_val, dt_mean, dt_sd = norm_dt(dt_train, dt_test, dt_val, obj_var)

    num_features = dt_train.shape[1]

    # Create the temporal window
    multi_window = WindowGenerator(input_width=input_width, label_width=out_steps, shift=out_steps,
                                   dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                                   label_columns=dt_train.columns)  # obj_var

    # Fit the model
    tmp = time.time()
    model = Conv1dnn(max_epochs, multi_window, num_features, model=model_arch, conv_width=conv_width, out_steps=out_steps)
    # model = AutoLSTM(MAX_EPOCHS, multi_window, num_features, units=32, out_steps=out_steps)
    model.train_net()
    model.fit_net(patience=patience)
    print("Elapsed training time: {:f} seconds".format(time.time() - tmp))

    # Forecasting
    if mode == 0:  # Simple prediction of the output
        res = model.predict(dt_test.iloc[ini:(ini + input_width + out_steps), :], obj_var=obj_var)

    elif mode == 1:  # Forecast of a certain length
        tmp = time.time()
        res = model.predict_long_term(dt_test.iloc[ini:(len(dt_test)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    else:  # Evaluate the MAE and MAPE of the model
        res = eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    return res, model

@multitasking.task
def main_pipeline_synth(dt, cv, idx_cyc, obj_var, ini, length, out_steps, units, input_width,
                        num_features, max_epochs, patience, model_arch=None, mode=3, single=False, queue=None):
    import tensorflow as tf
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.random.set_seed(424242)

    # Obtain the correspondent cycles from the dataset
    dt_train, dt_test, dt_val, cyc_idx_test = get_train_test_val(dt, cv['test'], cv['val'], idx_cyc)
    # cyc_idx_test = cyc_idx_test[1:]
    # dt_train = dt_train.diff(1)[1:]
    # dt_test = dt_test.diff(1)[1:]
    # dt_val = dt_val.diff(1)[1:]
    #dt_train, dt_test, dt_val, dt_mean, dt_sd = norm_dt(dt_train, dt_test, dt_val, obj_var)
    dt_train, dt_test, dt_val, dt_mean, dt_sd = norm_dt_min_max(dt_train, dt_test, dt_val, obj_var)

    model_arch = tf.keras.Sequential([
        tf.keras.layers.LSTM(units, return_sequences=False, #activation="swish",
                             kernel_initializer=tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)),
        #tf.keras.layers.Dropout(0.1),
        #tf.keras.layers.Dense(10),
        #tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(out_steps * num_features,
                              kernel_initializer=tf.initializers.zeros, activation="linear"),
        tf.keras.layers.Reshape([out_steps, num_features])])

    # model_arch = LSTM_rec(units, out_steps, num_features)

    #num_features = dt_train.shape[1]

    # Create the temporal window
    multi_window = WindowGenerator(input_width=input_width, label_width=out_steps, shift=out_steps,
                                   dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                                   label_columns=[obj_var])

    # Fit the model
    train_t = time.time()
    # model = Conv1dnn(max_epochs, multi_window, num_features, model=model_arch, conv_width=units, out_steps=out_steps)
    model = AutoLSTM(max_epochs, multi_window, num_features, model=model_arch, units=units, out_steps=out_steps)
    model.train_net()
    model.fit_net(patience=patience, show_plot=False)
    train_t = time.time() - train_t
    print("Elapsed training time: {:f} seconds".format(train_t))

    # Forecasting
    if mode == 0:  # Simple prediction of the output
        res = model.predict(dt_test.iloc[ini:(ini + input_width + out_steps), :], obj_var=obj_var)

    elif mode == 1:  # Forecast of a certain length
        tmp = time.time()
        res = model.predict_long_term(dt_test.iloc[ini:(len(dt_test)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    elif mode == 2:  # Evaluate the MAE and MAPE of the model
        res = eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    else:
        res = eval_test_rep(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd,
                            show_plot=False, single=single)

    if not(queue is None):
        queue.put([res[0], res[1], res[2], train_t])

    #return [res[0], res[1], train_t], model


@multitasking.task
def main_pipeline_stock(dt, cv, idx_cyc, obj_var, ini, length, out_steps, units, input_width,
                        num_features, max_epochs, patience, model_arch=None, mode=3, single=False, queue=None):
    import tensorflow as tf
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.random.set_seed(424242)

    # Obtain the correspondent cycles from the dataset
    dt_train, dt_test, dt_val, cyc_idx_test = get_train_test_val(dt, cv['test'], cv['val'], idx_cyc)
    dt_train, dt_test, dt_val, dt_mean, dt_sd = norm_dt_min_max(dt_train, dt_test, dt_val, obj_var)

    model_arch = tf.keras.Sequential([
        #tf.keras.layers.Lambda(lambda x: x[:, -units:, :]),
        tf.keras.layers.LSTM(units, return_sequences=True, #activation="swish",
                             kernel_initializer=tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units, return_sequences=True,
                             kernel_initializer=tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)),
        #tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(units, return_sequences=True,
                             kernel_initializer=tf.initializers.RandomUniform(minval=-0.1, maxval=0.1)),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(out_steps * num_features,
                              kernel_initializer=tf.initializers.zeros, activation="linear"),
        tf.keras.layers.Reshape([out_steps, num_features])])

    # Create the temporal window
    multi_window = WindowGenerator(input_width=input_width, label_width=out_steps, shift=out_steps,
                                   dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                                   label_columns=[obj_var])

    # Fit the model
    train_t = time.time()
    # model = Conv1dnn(max_epochs, multi_window, num_features, model=model_arch, conv_width=units, out_steps=out_steps)
    model = AutoLSTM(max_epochs, multi_window, num_features, model=model_arch, units=units, out_steps=out_steps)
    model.train_net()
    model.fit_net(patience=patience, show_plot=False)
    train_t = time.time() - train_t
    print("Elapsed training time: {:f} seconds".format(train_t))

    # Forecasting
    if mode == 0:  # Simple prediction of the output
        res = model.predict(dt_test.iloc[ini:(ini + input_width + out_steps), :], obj_var=obj_var)

    elif mode == 1:  # Forecast of a certain length
        tmp = time.time()
        res = model.predict_long_term(dt_test.iloc[ini:(len(dt_test)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    elif mode == 2:  # Evaluate the MAE and MAPE of the model
        res = eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    else:
        res = eval_test_rep(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd,
                            show_plot=False, single=single)

    if not(queue is None):
        queue.put([res[0], res[1], res[2], train_t])