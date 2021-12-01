import tensorflow as tf
from .net_factory import TsNetwork
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy import zeros, append

class AutoLSTM(TsNetwork):
    def __init__(self, epochs, window, num_features, model=None, units=32, out_steps=50):
        super().__init__(epochs, window, model, out_steps)
        self._num_features = num_features
        self._units = units
        if self._model is None:
            self._model = self._default_model()

    # Obsolete?
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    # Define the loss, optimizer and metrics and compile the model
    def train_net(self, loss=tf.losses.MeanSquaredError(), opt=tf.optimizers.Adam(learning_rate=0.001),
                  metrics=[tf.metrics.MeanAbsoluteError()]):
        self._model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Fit the keras model to the training data and validate the results
    def fit_net(self, patience=5, show_plot=True):
        if patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience, mode='min')
            log = self._model.fit(self._window.train, epochs=self._epochs, validation_data=self._window.val,
                                  callbacks=[early_stopping])
        else:
            log = self._model.fit(self._window.train, epochs=self._epochs, validation_data=self._window.val)
        if show_plot:
            self._plot_train_val_loss(log)

        return log

    def predict(self, dt, obj_var, show_plot=True):
        if len(dt) < self._window.input_width:
            raise ValueError(f'At least {self._window.input_width} previous instants have to be provided')
        elif len(dt) < self._window.total_window_size:  # If we only have the input values, the rest of the df is empty padding
            dt = dt.iloc[0:self._window.input_width, :]
            dt_empty = DataFrame(None, index=range(self._window.label_width), columns=dt.columns)
            dt = dt.append(dt_empty)
            del dt_empty

        prep_dt = self._window.make_dataset(dt)
        inputs, labels = next(iter(prep_dt))
        preds = self._model.predict(inputs)

        if show_plot:
            self._plot_predictions(inputs, labels, preds, obj_var)

        return preds

    def predict_single_shot(self, dt, obj_var, length, show_plot=True):
        if len(dt) < self._window.input_width:
            raise ValueError(f'At least {self._window.input_width} previous instants have to be provided')

        dt_ini = dt.iloc[0:self._window.input_width, :]
        dt_empty = DataFrame(None, index=range(self._window.label_width), columns=dt_ini.columns)
        dt_ini = dt_ini.append(dt_empty)

        prep_dt = self._window.make_dataset(dt_ini)
        inputs, labels = next(iter(prep_dt))
        preds = self._model.predict(inputs)
        path = preds[0, :]

        if show_plot:
            self._plot_predictions_long_term(dt, path, obj_var, length)

        return path[:, self._window.label_columns_indices.get(obj_var)]

    def predict_long_term(self, dt, obj_var, length, show_plot=True):
        if len(dt) < self._window.input_width:
            raise ValueError(f'At least {self._window.input_width} previous instants have to be provided')

        dt_ini = dt.iloc[0:self._window.input_width, :]
        dt_empty = DataFrame(None, index=range(self._window.label_width), columns=dt_ini.columns)
        dt_ini = dt_ini.append(dt_empty)

        iterations = -(-length // self._window.label_width)  # Number of recurrent steps. Ceiling of the division
        path = zeros((0, self._num_features))
        for i in range(iterations):
            prep_dt = self._window.make_dataset(dt_ini)
            inputs, labels = next(iter(prep_dt))
            preds = self._model.predict(inputs)
            path = append(path, preds[0, :, :], axis=0)
            if self._window.label_width > self._window.input_width:
                dt_ini.iloc[0:self._window.input_width, :] = \
                    preds[0, range(self._window.label_width - self._window.input_width,
                                   self._window.label_width), :]  # Move the predictions as evidence
            elif self._window.label_width < self._window.input_width:
                dt_ini.iloc[0:self._window.input_width-self._window.label_width, :] = \
                    dt_ini.iloc[self._window.input_width - self._window.label_width:self._window.input_width, :]
                dt_ini.iloc[self._window.input_width - self._window.label_width:self._window.input_width, :] = preds[0, :, :]
            else:
                dt_ini.iloc[0:self._window.input_width, :] = preds[0, :, :]

            dt_ini.iloc[0:self._window.input_width, :] = \
                preds[0, range(self._window.label_width - self._window.input_width,
                               self._window.label_width), :]  # Move the predictions as evidence

        if show_plot:
            self._plot_predictions_long_term(dt, path, obj_var, length)

        return path

    def del_model(self):
        self._model = None

    def _default_model(self):
        return tf.keras.Sequential([
            # Shape [batch, time, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            tf.keras.layers.LSTM(self._units),
            tf.keras.layers.Dense(self._out_steps * self._num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([self._out_steps, self._num_features])])

    def _plot_predictions(self, inputs, labels, preds, obj_var):
        plt.figure()
        initial_line, = plt.plot(range(self._window.input_width),
                                 inputs[0, :, self._window.column_indices.get(obj_var)], label='Initial values')
        real_line, = plt.plot(range(self._window.input_width - 1, self._window.total_window_size - 1),
                              labels[0, :, self._window.label_columns_indices.get(obj_var)], label='Real values')
        pred_line, = plt.plot(range(self._window.input_width - 1, self._window.total_window_size - 1),
                              preds[0, :, self._window.label_columns_indices.get(obj_var)],
                              label='Predicted values')
        plt.legend(handles=[initial_line, real_line, pred_line])
        plt.ylabel(obj_var)
        plt.xlabel('Time (h)')
        plt.show()

    def _plot_predictions_long_term(self, dt, path, obj_var, length):
        plt.figure()
        initial_line, = plt.plot(range(self._window.input_width),
                                 dt[obj_var][0:self._window.input_width], label='Initial values')
        real_line, = plt.plot(range(self._window.input_width - 1, length + self._window.input_width - 1),
                              dt[obj_var][self._window.input_width:(length + self._window.input_width)],
                              label='Real values')
        pred_line, = plt.plot(range(self._window.input_width - 1, length + self._window.input_width - 1),
                              path[0:length, self._window.label_columns_indices.get(obj_var)], label='Predicted values')
        plt.legend(handles=[initial_line, real_line, pred_line])
        plt.ylabel(obj_var)
        plt.xlabel('Time (h)')
        plt.show()

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):

        return None
