from abc import ABC, abstractmethod


class TsNetwork(ABC):

    def __init__(self, epochs, batch):
        self._epochs = epochs
        self._batch = batch
        self._model = None

    @abstractmethod
    def train_net(self, x_train, y_train):
        pass

    @abstractmethod
    def fit_net(self, x_train, y_train, x_test, y_test):
        pass

    @abstractmethod
    def predict(self, x_test, y_test, verbose):
        pass

    @abstractmethod
    def predict_long_term(self, x_test, y_test, verbose):
        pass
