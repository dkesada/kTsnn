from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.optimizers import RMSprop, sgd, adam
import pandas as pd
from kTsnn.src.utils import *
from .net_factory import TsNetwork


class Conv1dnn(TsNetwork):
    def __init__(self, epochs, batch):
        super().__init__(epochs, batch)

    # Define the structure of the 2dCNN and compile the model
    # Not parametrized for now, changes are hard coded inside
    # I should resort to some structure-storing keras method
    def train_net(self, x_train, y_train):
        inputs = Input(shape=(x_train.shape[1], x_train.shape[2]))
        x = Conv1D(filters=64, kernel_size=2, activation='relu')(inputs)
        x = MaxPooling1D(pool_size=1)(x)
        x = Flatten()(x)
        x = Dense(50, activation='relu')(x)
        pred = Dense(pd.DataFrame(y_train).shape[1])(x)
        self._model = Model(inputs=inputs, outputs=pred)
        optim = adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self._model.compile(loss='mae', optimizer=optim, metrics=['mae'])

        return None

    # Fit the keras model to the training data and validate the results
    # This time around, the pandas df has to be translated to numpy
    def fit_net(self, x_train, y_train, x_test, y_test):
        log = self._model.fit(x_train, y_train, batch_size=self._batch,
                              epochs=self._epochs, validation_data=(x_test, y_test))
        plot_train_val_loss(log)
        return log


    def predict(self, x_test, y_test, verbose):
        return self._model.predict(x_test)

    # Function to do long term forecasting with a trained TDNN
    def predict_long_term(self, x_test, y_test, obj_var):

        return None

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):

        return None