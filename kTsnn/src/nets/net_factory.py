from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class TsNetwork(ABC):
    """
    Interface for different time-series neural networks

    Attributes
    ----------

    epochs: int
        number of epochs for training
    window: WindowGenerator
        window with the training input_width, label_width and shift desired
    model: tensorflow.keras.Sequential
        structure of the model in case we want a different one from the default

    Methods
    -------

    train_net(x_train, y_train, **kwargs)
        trains the NN with its own implemented method
    fit_net(x_train, y_train, x_test, y_test, **kwargs)
        fits the NN with its own implemented method
    predict(x_test, y_test, **kwargs)
        predicts a dataset with the model
    predict_long_term(x_test, y_test, obj_var, ini, length, **kwargs)
        predicts a time-series in the long term

    """

    def __init__(self, epochs, window, model, out_steps, **kwargs):
        self._epochs = epochs
        self._window = window
        self._model = model
        self._out_steps = out_steps

    def get_input_width(self):
        return self._window.input_width

    def get_num_labels(self):
        return len(self._window.label_columns)

    def get_out_steps(self):
        return self._out_steps

    @abstractmethod
    def train_net(self, loss, optimizer, metric, **kwargs):
        pass

    @abstractmethod
    def fit_net(self, patience, **kwargs):
        pass

    @abstractmethod
    def predict(self, dt, obj_var, show_plot):
        pass

    @abstractmethod
    def predict_long_term(self, dt, obj_var, ini, length, **kwargs):
        pass

    @staticmethod
    def _plot_train_val_loss(log):
        plt.figure()
        train_line, = plt.plot(log.history['loss'], label='Train loss')
        val_line, = plt.plot(log.history['val_loss'], label='Validation loss')
        plt.legend(handles=[train_line, val_line])
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    @abstractmethod
    def _default_model(self, **kwargs):
        pass
