from utils import *
from nets.tdnn import TDNN
import pandas as pd
import os

dt_file = 'prep_data_fold_2.csv'
info_file = 'exec_info.txt'

if __name__ == '__main__':
    dt = load_dt(dt_file)
    info = load_info(info_file)
    epochs = 20
    batch = 10
    dt_test, y_test, dt_train, y_train = get_train_test(dt, info['idx_test'], info['obj_var'], info['idx_cyc'])
    net = TDNN(epochs, batch)
    net.train_net(dt_train, y_train)
    net.fit_net(dt_train, y_train, dt_test, y_test)
