from abc import ABC, abstractmethod


class TsNetwork(ABC):
    """
    Interface for different time-series neural networks

    Attributes
    ----------

    epochs: int
        number of epochs for training
    batch: int
        size of the batch

    Methods
    -------

    train_net(x_train, y_train, **kwargs)
        trains the NN with its own implemented method
    fit_net(x_train, y_train, x_test, y_test, **kwargs)
        fits the NN with its own implemented method
    train_net(x_train, y_train, **kwargs)
        trains the NN with its own implemented methods
    predict(x_test, y_test, **kwargs)
        predicts a dataset with the model
    predict_long_term(x_test, y_test, obj_var, ini, length, **kwargs)
        predicts a time-series in the long term

    """

    def __init__(self, epochs, batch, **kwargs):
        self._epochs = epochs
        self._batch = batch
        self._model = None

    @abstractmethod
    def train_net(self, x_train, y_train, **kwargs):
        pass

    @abstractmethod
    def fit_net(self, x_train, y_train, x_test, y_test, **kwargs):
        pass

    @abstractmethod
    def predict(self, x_test, y_test, **kwargs):
        pass

    @abstractmethod
    def predict_long_term(self, x_test, y_test, obj_var, ini, length, **kwargs):
        pass
