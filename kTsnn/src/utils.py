import plotly.express as plot
from numpy import array, hstack, vstack

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


def split_row(df, row, size):
    ret = array(df[grep_columns(df, 't_1')].iloc[row])
    for i in range(2, size+1):
        ret = vstack((ret, array(df[grep_columns(df, 't_' + str(i))].iloc[row])))

    return ret

# Converts a pandas dataframe into a numpy one
def zip_df(df, size):
    if size == 1:
        return array(df)

    ret = array([])
    ret = ret.reshape(0, size, int(df.shape[1] / size))
    for i in range(df.shape[0]):
        ret = vstack((ret, array([split_row(df, i, size)])))

    return ret


# Return the x and y dataframes for both train and validation in numpy arrays
# It's bloody slow. Rather badly coded
def get_train_test_np(dt, cv, obj_col, cyc_col):
    dt_test, y_test, dt_train, y_train = get_train_test(dt, cv, obj_col, cyc_col)
    size = int(dt_test.shape[1] / y_test.shape[1])

    return zip_df(dt_test, size), array(y_test), zip_df(dt_train, size), array(y_train)