from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import RMSprop, sgd, adam
import pandas as pd
import plotly.express as plot
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
def get_train_test(dt, cv, obj_col, cyc_col):
    in_test = map_w(lambda x: x in cv, dt[cyc_col])
    dt_test = dt[in_test]
    y_test = dt_test[obj_col]
    dt_train = dt[map_w(lambda x: not x, in_test)]
    y_train = dt_train[obj_col]
    dt_test = dt_test.drop(columns=[cyc_col] + obj_col)
    dt_train = dt_train.drop(columns=[cyc_col] + obj_col)

    return dt_test, y_test, dt_train, y_train


# Define the structure of the TDNN and compile the model
# Not parametrized for now, changes are hard coded inside
def train_tdnn(dt_train, y_train):
    inputs = Input(shape=dt_train.shape[1:])
    x = Dense(64, activation='relu')(inputs)
    #x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    #x = Dense(32, activation='relu')(x)
    #x = Dense(16, activation='relu')(x)
    #x = Dense(8, activation='relu')(x)
    pred = Dense(pd.DataFrame(y_train).shape[1])(x)
    model = Model(inputs=inputs, outputs=pred)
    optim = adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss='mae', optimizer=optim, metrics=['mae'])

    return model


# Fit the keras model to the training data and validate the results
def fit_tdnn(dt_train, y_train, dt_test, y_test, model, epochs = 3, batch = 32):
    log = model.fit(dt_train, y_train, batch_size=batch, epochs=epochs, validation_data=(dt_test, y_test))
    plot_train_val_loss(log)
    return model


# Main function that splits the data and trains a model. Can be used with any dt_fold size
def train_and_fit_tdnn(dt, cv, obj_col, cyc_col, epochs = 5, batch = 32):
    dt_test, y_test, dt_train, y_train = get_train_test(dt, cv, obj_col=obj_col, cyc_col=cyc_col)
    model = train_tdnn(dt_train, y_train)
    model = fit_tdnn(dt_train, y_train, dt_test, y_test, model, epochs, batch)

    return model


# Find the columns that have the given string in their names
def grep_columns(dt, sub):
    return dt.columns[map_w(lambda x: sub in x, dt.columns)]


# Move the evidence one time slice and introduce the new predictions as evidence
def move_evidence(evidence, particles):
    size = evidence.shape[1] / particles.shape[1]
    evidence = evidence.copy()  # With love, SettingWithCopyWarning
    for i in range(1, int(size)):
        evidence.loc[:, grep_columns(evidence, 't_' + str(i+1))] =\
            evidence[grep_columns(evidence, 't_' + str(i))].values.flatten()

    evidence.loc[:, grep_columns(evidence, 't_1')] = particles

    return evidence


# Function to do long term forecasting with a trained TDNN
def predict_long_term(dt_test, y_test, model, obj_var):
    path = []
    evidence = dt_test.iloc[0:1]
    for i in range(dt_test.shape[0]):
        particles = model.predict(evidence)
        obj_idx = list(y_test.columns).index(obj_var)
        path = path + [particles[0, obj_idx]]
        evidence = move_evidence(evidence, particles)

    return path


# Script part that gets the data, trains a model and plots the results

# JSON with the path to the data and the names of the variables and the columns
with open('aux_data.json', 'r') as json_file:
    aux_data = json.load(json_file)

dt = pd.read_csv(aux_data['dt_path'])
dt_f = dt[dt[aux_data['cycles_col']] != 18]
cv = [[17, 5, 1, 10], [6, 3, 13, 2], [15, 14, 9, 16], [12, 19, 11, 7], [8, 20, 4, 21]]

obj_cols = dt_f.columns
obj_cols = obj_cols[map_w(lambda x: 't_0' in x, obj_cols)]  # The dataframe was folded in R with my "dbnR" package
obj_cols = list(obj_cols.drop(aux_data['cycles_col']))

dt_test, y_test, _, _ = get_train_test(dt_f, cv[0], obj_col=obj_cols, cyc_col=aux_data['cycles_col'])

# Train and predict
tdnn = train_and_fit_tdnn(dt_f, cv[0], epochs=150, batch=100, obj_col=obj_cols, cyc_col=aux_data['cycles_col'])
pred = tdnn.predict(dt_test)
obj_idx = list(y_test.columns).index(aux_data['obj_var'])

# Plot of the predictions
fig = plot.line()
fig = fig.add_scatter(y=pred[:, obj_idx].flatten(), mode='lines', name='Prediction')
fig = fig.add_scatter(y=y_test[aux_data['obj_var']], mode='lines', name='Reality')
fig.show()

# Long term forecasting
lpred = predict_long_term(dt_test.iloc[0:1000], y_test, tdnn, obj_var=aux_data['obj_var'])

# Plot of the predictions
fig = plot.line()
fig = fig.add_scatter(y=lpred, mode='lines', name='Prediction')
fig = fig.add_scatter(y=y_test.iloc[0:1000][aux_data['obj_var']], mode='lines', name='Reality')
fig.show()
