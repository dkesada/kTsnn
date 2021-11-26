import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import numpy as np
import time
from kTsnn.src.nets.cnn import Conv1dnn
from kTsnn.src.nets.lstm import AutoLSTM
from kTsnn.src.nets.window_gen import WindowGenerator


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


def mae(orig, pred):
    tmp = pred - orig
    tmp = np.abs(tmp)
    return tmp.sum() / len(tmp)


def undo_norm(ts, mean, sd):
    return ts * sd + mean


def eval_pred(orig, pred, mean, sd):
    if(len(orig) > len(pred)):
        orig = orig[0:len(pred)]
    orig_un = undo_norm(orig, mean, sd)
    pred_un = undo_norm(pred, mean, sd)
    res = mae(orig_un, pred_un)
    print("MAE: {:f}".format(res))

    return res


def eval_model(model, dt_test, ini, length, obj_var, mean, sd, show_plot=False):
    obj_idx = dt_test.columns.get_loc(obj_var)
    orig = dt_test[obj_var]
    path = model.predict_long_term(dt_test.iloc[ini:(ini+length), :],
                                   obj_var=obj_var, length=length, show_plot=show_plot)
    res = eval_pred(orig, path[:, obj_idx], mean, sd)

    return res


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
    res = np.array([zeros(n_preds), zeros(n_preds)])
    for i in range(n_preds):
        tmp = time.time()
        path = self.predict_long_term(dt.iloc[ini:(len(dt)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    return (res)

# In case we do more than one forecasting per cycle
def eval_test_rep(model, dt_test, cyc_idx_test, ini, length, obj_var, mean, sd, show_plot=False):
    res = np.array([np.zeros(len(cyc_idx_test.unique())), np.zeros(len(cyc_idx_test.unique()))])
    j = 0
    pad_size = model.get_input_width() + length
    for i in cyc_idx_test.unique():
        rows = cyc_idx_test == i
        n_preds = int(((dt_test[rows].shape[0] - ini) / pad_size) // 1)
        cyc_res = np.array([np.zeros(n_preds), np.zeros(n_preds)])
        for k in range(n_preds):
            tmp = time.time()
            cyc_res[0][k] = eval_model(model, dt_test[rows], ini + k * pad_size, length, obj_var, mean, sd, show_plot)
            cyc_res[1][k] = time.time() - tmp
            print("Elapsed forecasting time: {:f} seconds".format(res[1][j]))

        res[0][j] = cyc_res[0].mean()
        res[1][j] = cyc_res[1].mean()
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

    else:  # Evaluate the MAE of the model
        res = eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    return res, model


def main_pipeline_synth(dt, cv, idx_cyc, obj_var, ini, length, out_steps, units, input_width,
                        max_epochs, patience, model_arch=None, mode=3):
    # Obtain the correspondent cycles from the dataset
    dt_train, dt_test, dt_val, cyc_idx_test = get_train_test_val(dt, cv['test'], cv['val'], idx_cyc)
    dt_train, dt_test, dt_val, dt_mean, dt_sd = norm_dt(dt_train, dt_test, dt_val, obj_var)

    num_features = dt_train.shape[1]

    # Create the temporal window
    multi_window = WindowGenerator(input_width=input_width, label_width=out_steps, shift=out_steps,
                                   dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                                   label_columns=[obj_var])  # obj_var,

    # Fit the model
    train_t = time.time()
    #model = Conv1dnn(max_epochs, multi_window, num_features, model=model_arch, conv_width=conv_width, out_steps=out_steps)
    model = AutoLSTM(max_epochs, multi_window, num_features, model=model_arch, units=units, out_steps=out_steps)
    model.train_net()
    model.fit_net(patience=patience)
    train_t = time.time() - train_t
    print("Elapsed training time: {:f} seconds".format(train_t))

    # Forecasting
    if mode == 0:  # Simple prediction of the output
        res = model.predict(dt_test.iloc[ini:(ini + input_width + out_steps), :], obj_var=obj_var)

    elif mode == 1:  # Forecast of a certain length
        tmp = time.time()
        res = model.predict_long_term(dt_test.iloc[ini:(len(dt_test)), :], obj_var=obj_var, length=length)
        print("Elapsed forecasting time: {:f} seconds".format(time.time() - tmp))

    elif mode == 2:  # Evaluate the MAE of the model
        res = eval_test(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    else:
        res = eval_test_rep(model, dt_test, cyc_idx_test, ini, length, obj_var, dt_mean, dt_sd)

    return [res[0], res[1], train_t], model

