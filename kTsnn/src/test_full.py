import pandas as pd
import json
import os
import plotly.express as plot
from kTsnn.src.nets.tdnn import TDNN
from kTsnn.src.utils import get_train_test, map_w, load_dt, load_info

# Script part that gets the data, trains a model and plots the results

# JSON with the names of the variables and the columns
dt_file = 'prep_data_fold_2.csv'
info_file = 'exec_info.txt'

if __name__ == '__main__':
    dt = load_dt(dt_file)
    info = load_info(info_file)

    dt = dt[dt[info['idx_cyc']] != 18]

    x_test, y_test, x_train, y_train = get_train_test(dt, info['cv'][0], obj_col=info['obj_cols'], cyc_col=info['idx_cyc'])

    # Train and predict
    tdnn = TDNN(epochs=50, batch=100)
    tdnn.train_net(x_train, y_train)
    tdnn.fit_net(x_train, y_train, x_test, y_test)
    pred = tdnn.predict(x_test, y_test)

    # Plot of the predictions
    obj_var_idx = info['obj_cols'].index(info['obj_var'])

    fig = plot.line()
    fig = fig.add_scatter(y=pred[:, obj_var_idx].flatten(), mode='lines', name='Prediction')
    fig = fig.add_scatter(y=y_test[info['obj_var']], mode='lines', name='Reality')
    fig.show()

    # Long term forecasting
    lpred = tdnn.predict_long_term(x_test.iloc[0:100], y_test.iloc[0:100], obj_var=info['obj_var'], ini=0, length=100)

    # Plot of the predictions
    fig = plot.line()
    fig = fig.add_scatter(y=lpred, mode='lines', name='Prediction')
    fig = fig.add_scatter(y=y_test.iloc[0:100][info['obj_var']], mode='lines', name='Reality')
    fig.show()