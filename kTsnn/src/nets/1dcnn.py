from keras.models import Model
from keras.layers import Dense, Input, Dropout, Conv2D, Conv1D, Flatten
from keras.optimizers import RMSprop, sgd, adam
import pandas as pd
from numpy import array, hstack
from kTsnn.src.utils import *
from .net_factory import TsNetwork


class Conv1dnn(TsNetwork):
    def __init__(self, epochs, batch):
        super().__init__(epochs, batch)

    # Define the structure of the 2dCNN and compile the model
    # Not parametrized for now, changes are hard coded inside
    # I should resort to some structure-storing keras method
    def train_net(self, x_train, y_train):

        return None

    # Fit the keras model to the training data and validate the results
    # This time around, the pandas df has to be translated to numpy
    def fit_net(self, x_train, y_train, x_test, y_test):

        return None

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):

        return None

    def predict(self, x_test, y_test, verbose):
        return None

    # Function to do long term forecasting with a trained TDNN
    def predict_long_term(self, x_test, y_test, obj_var):

        return None
