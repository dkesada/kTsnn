from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, Conv1D, Flatten
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
import pandas as pd
from kTsnn.src.utils import *
from .net_factory import TsNetwork


class TDNN(TsNetwork):
    def __init__(self, epochs, batch):
        super().__init__(epochs, batch)

    # Define the structure of the TDNN and compile the model
    # Not parametrized for now, changes are hard coded inside
    # I should resort to some structure-storing keras method
    def train_net(self, x_train, y_train, **kwargs):
        inputs = Input(shape=x_train.shape[1:])
        x = Dense(528, activation='relu')(inputs)
        x = Dense(274, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        #x = Dense(32, activation='relu')(x)
        #x = Dense(16, activation='relu')(x)
        #x = Dense(8, activation='relu')(x)
        pred = Dense(pd.DataFrame(y_train).shape[1])(x)
        self._model = Model(inputs=inputs, outputs=pred)
        optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        self._model.compile(loss='mae', optimizer=optim, metrics=['mae'])

    # Fit the keras model to the training data and validate the results
    def fit_net(self, x_train, y_train, x_test, y_test, **kwargs):
        log = self._model.fit(x_train, y_train, batch_size=self._batch,
                              epochs=self._epochs, validation_data=(x_test, y_test))
        plot_train_val_loss(log)
        return log

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):
        size = evidence.shape[1] / particles.shape[1]
        evidence = evidence.copy()  # With love, SettingWithCopyWarning
        for i in range(1, int(size)):
            evidence.loc[:, grep_columns(evidence, 't_' + str(i+1))] =\
                evidence[grep_columns(evidence, 't_' + str(i))].values.flatten()

        evidence.loc[:, grep_columns(evidence, 't_1')] = particles

        return evidence

    def predict(self, x_test, y_test, **kwargs):
        return self._model.predict(x_test)

    # Function to do long term forecasting with a trained TDNN
    def predict_long_term(self, x_test, y_test, obj_var, ini, length, **kwargs):
        path = []
        evidence = x_test.iloc[ini:ini+1]
        obj_idx = list(y_test.columns).index(obj_var)
        for i in range(ini, ini + length):
            particles = self._model.predict(evidence)
            path = path + [particles[0, obj_idx]]
            evidence = self.__move_evidence(evidence, particles)

        return path


