import tensorflow as tf
from .net_factory import TsNetwork
from pandas import DataFrame
import matplotlib.pyplot as plt


class AutoLSTM(TsNetwork):
    def __init__(self, epochs, window, num_features, model=None, units=32, out_steps=50):
        self._num_features = num_features
        self._units = units
        self._out_steps = out_steps
        if model is None:
            model = self._default_model()
        super().__init__(epochs, window, model)

        # Obsolete?
        # TsNetwork.__init__(self, epochs, window, None)
        # tf.keras.Model.__init__(self)
        # self.out_steps = out_steps
        # self._num_features = num_features
        # self.units = units
        # self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        # self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        # self.dense = tf.keras.layers.Dense(num_features)

    # Obsolete?
    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    # Define the loss, optimizer and metrics and compile the model
    def train_net(self, loss=tf.losses.MeanSquaredError(), opt=tf.optimizers.Adam(learning_rate=0.0001, ),
                  metrics=[tf.metrics.MeanAbsoluteError()]):
        self._model.compile(loss=loss, optimizer=opt, metrics=metrics)

    # Fit the keras model to the training data and validate the results
    def fit_net(self, patience=5, show_plot=True):
        if patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='min')
            log = self.fit(self._window.train, epochs=self._epochs, validation_data=self._window.val,
                                  callbacks=[early_stopping])
        else:
            log = self.fit(self._window.train, epochs=self._epochs, validation_data=self._window.val)
        if show_plot:
            self._plot_train_val_loss(log)

        return log

    def predict(self, dt, obj_var, show_plot):
        return None

    def predict_long_term(self, dt, obj_var, length, show_plot=True):
        if len(dt) < self._window.input_width:
            raise ValueError(f'At least {self._window.input_width} previous instants have to be provided')

        dt_ini = dt.iloc[0:self._window.input_width, :]
        dt_empty = DataFrame(None, index=range(self._window.label_width), columns=dt_ini.columns)
        dt_ini = dt_ini.append(dt_empty)
        prep_dt = self._window.make_dataset(dt_ini)
        inputs, labels = next(iter(prep_dt))

        return self.call(inputs, None)

    # Function to do long term forecasting with a trained TDNN
    def call(self, inputs, training):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])

        return predictions

    def _default_model(self):
        return tf.keras.Sequential([
            # Shape [batch, time, features]
            tf.keras.layers.LSTM(self._units, batch_input_shape=)
            tf.keras.layers.Lambda(lambda x: x[:, -self._conv_width:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=self._conv_width),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(self._out_steps * self._num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
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
                              dt[obj_var][range(self._window.input_width, length+self._window.input_width)], label='Real values')
        pred_line, = plt.plot(range(self._window.input_width - 1, length),
                              path[range(self._window.input_width-1, length),
                                   self._window.label_columns_indices.get(obj_var)],label='Predicted values')
        plt.legend(handles=[initial_line, real_line, pred_line])
        plt.ylabel(obj_var)
        plt.xlabel('Time (h)')
        plt.show()

    # Move the evidence one time slice and introduce the new predictions as evidence
    @staticmethod
    def __move_evidence(evidence, particles):

        return None
