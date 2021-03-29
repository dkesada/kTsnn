import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import numpy as np


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
def norm_dt(dt_train, dt_test, dt_val):
    dt_mean = dt_train.mean()
    dt_sd = dt_train.std()
    dt_train = (dt_train - dt_mean) / dt_sd
    dt_test = (dt_test - dt_mean) / dt_sd
    dt_val = (dt_val - dt_mean) / dt_sd
    dt_mean = dt_mean[dt_train.columns.get_loc(info['obj_var'][0])]
    dt_sd = dt_sd[dt_train.columns.get_loc(info['obj_var'][0])]

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
        res[j] = eval_model(model, dt_test[rows], ini, length, obj_var, mean, sd, show_plot)
        j += 1

    return res


