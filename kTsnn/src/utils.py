import plotly.express as plot
import os
import pandas as pd
import json

# Line plot of a column in a dataframe
def plot_col(dt, col):
    plot.line(dt, y=col).show()


# Line plot of the loss in each epoch in train and validation
def plot_train_val_loss(log):
    fig = plot.line()
    fig = fig.add_scatter(y=log.history['loss'], mode='lines', name='Train loss')
    fig = fig.add_scatter(y=log.history['val_loss'], mode='lines', name='Validation loss')
    fig.update_layout(xaxis_title='Epochs', yaxis_title='Loss')
    fig.show()


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
    dt_train = dt_train.drop(columns=cyc_col)
    dt_test = dt_test.drop(columns=cyc_col)
    dt_val = dt_val.drop(columns=cyc_col)

    return dt_train, dt_test, dt_val

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
    dt_std = dt_train.std()
    dt_train = (dt_train - dt_mean) / dt_std
    dt_test = (dt_test - dt_mean) / dt_std
    dt_val = (dt_val - dt_mean) / dt_std

    return dt_train, dt_test, dt_val



