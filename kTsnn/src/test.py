from kTsnn.src.utils import *
from kTsnn.src.nets.tdnn import TDNN
from kTsnn.src.nets.transition import Transition
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 dt_train, dt_test, dt_val,
                 label_columns=None):
        # Store the raw data.
        self.dt_train = dt_train
        self.dt_val = dt_val
        self.dt_test = dt_test

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(dt_train.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, plot_col, model=None, max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32, )

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.dt_train)

    @property
    def test(self):
        return self.make_dataset(self.dt_test)

    @property
    def val(self):
        return self.make_dataset(self.dt_val)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result



dt_file = 'dt_unfolded.csv'
info_file = 'exec_info.txt'

if __name__ == '__main__':
    dt = load_dt(dt_file)
    info = load_info(info_file)
    epochs = 20
    batch = 10
    num_features = dt.shape[1]


    dt_train, dt_test, dt_val = get_train_test_val(dt, info['test'], info['val'], info['idx_cyc'])
    #dt_train, dt_test, dt_val = norm_dt(dt_train, dt_test, dt_val)

    w1 = WindowGenerator(input_width=48, label_width=1, shift=24,
                         dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                         label_columns=info['obj_var'])

    single_step_window = WindowGenerator(input_width=1, label_width=1, shift=1,
                                         dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                                         label_columns=info['obj_var'])

    # Transition model
    model = Transition(label_index=single_step_window.column_indices[info['obj_var'][0]])
    model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])

    val_p = {}
    p = {}
    val_p['Transition'] = model.evaluate(single_step_window.val)
    p['Transition'] = model.evaluate(single_step_window.test, verbose=0)

    window2 = WindowGenerator(input_width=24, label_width=24, shift=1,
                              dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                              label_columns=info['obj_var'])

    #window2.plot(plot_col=info['obj_var'][0], model=model)

    # Generic compile
    MAX_EPOCHS = 20

    def compile_and_fit(model, window, patience=5):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history


    # Linear model
    # linear = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    # history = compile_and_fit(linear, single_step_window)
    #
    # val_p['Linear'] = linear.evaluate(single_step_window.val)
    # p['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    #window2.plot(plot_col=info['obj_var'][0], model=linear)

    # Dense model
    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=264, activation='relu'),
    #     tf.keras.layers.Dense(units=164, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])
    #
    # history = compile_and_fit(dense, single_step_window)
    #
    # val_p['Dense'] = dense.evaluate(single_step_window.val)
    # p['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
    # window2.plot(plot_col=info['obj_var'][0], model=dense)

    # Convolutional
    # LABEL_WIDTH=24
    # CONV_WIDTH = 10
    # INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    # conv_window = WindowGenerator(input_width=CONV_WIDTH, label_width=1, shift=1,
    #                                    dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
    #                                    label_columns=info['obj_var'])
    # wide_conv_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1,
    #                      dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
    #                      label_columns=info['obj_var'])
    #
    #
    # conv_model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=32,
    #                            kernel_size=(CONV_WIDTH,),
    #                            activation='relu'),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    # ])
    #
    # print('Input shape:', conv_window.example[0].shape)
    # print('Output shape:', conv_model(conv_window.example[0]).shape)
    #
    # history = compile_and_fit(conv_model, conv_window)
    # val_p['Conv'] = conv_model.evaluate(conv_window.val)
    # p['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
    # wide_conv_window.plot(plot_col=info['obj_var'][0], model=conv_model)

    # Forecasting into the future
    OUT_STEPS = 50
    multi_window = WindowGenerator(input_width=7, label_width=OUT_STEPS, shift=OUT_STEPS,
                         dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                         label_columns=info['obj_var'])
    CONV_WIDTH = 7
    multi_conv_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        # Shape => [batch, 1, conv_units]
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
        # Shape => [batch, 1,  out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    history = compile_and_fit(multi_conv_model, multi_window)

    val_p['Conv'] = multi_conv_model.evaluate(multi_window.val)
    p['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    multi_window.plot(plot_col=info['obj_var'][0], model=multi_conv_model)
