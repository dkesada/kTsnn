import plotly.express as plot

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