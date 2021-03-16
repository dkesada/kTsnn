import tensorflow as tf
from kTsnn.src.utils import *
from .net_factory import TsNetwork
from matplotlib.pyplot import plot as plt
from matplotlib.pyplot import cla


class Conv1dnn(TsNetwork):
    def __init__(self, epochs, window, num_features, model=None, conv_width=7, out_steps=50):
        self._num_features = num_features
        self._conv_width = conv_width
        self._out_steps = out_steps
        if model is None:
            model = self._default_model()
        super().__init__(epochs, window, model)

    # Define the loss, optimizer and metrics and compile the model
    def train_net(self, loss=tf.losses.MeanSquaredError(), opt=tf.optimizers.Adam(), metrics=[tf.metrics.MeanAbsoluteError()]):
        self._model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Fit the keras model to the training data and validate the results
    def fit_net(self, patience=5):
        early_stopping = None
        if patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')
        log = self._model.fit(self._window.train, epochs=self._epochs, validation_data=self._window.val,
                              callbacks=[early_stopping])
        plot_train_val_loss(log)

        return log

    def predict(self, dt):
        return self._model.predict(dt)

    # Function to do long term forecasting with a trained TDNN
    def predict_long_term(self, x_test, y_test, obj_var):

        return None

    def _default_model(self):
        return tf.keras.Sequential([
                    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                    tf.keras.layers.Lambda(lambda x: x[:, -self._conv_width:, :]),
                    # Shape => [batch, 1, conv_units]
                    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=self._conv_width),
                    # Shape => [batch, 1,  out_steps*features]
                    tf.keras.layers.Dense(self._out_steps * self._num_features,
                                          kernel_initializer=tf.initializers.zeros),
                    # Shape => [batch, out_steps, features]
                    tf.keras.layers.Reshape([self._out_steps, self._num_features])])

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):

        return None
