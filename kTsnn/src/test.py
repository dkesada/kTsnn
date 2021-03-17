from kTsnn.src.utils import *
from kTsnn.src.nets.cnn import Conv1dnn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kTsnn.src.nets.window_gen import WindowGenerator


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

    val_p = {}
    p = {}

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

    model = Conv1dnn(MAX_EPOCHS, multi_window, num_features, conv_width=CONV_WIDTH)
    model.train_net()
    model.fit_net()
    pred = model.predict(dt_test.loc[0:57, :], obj_var=info['obj_var'][0])

    # multi_conv_model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    #     # Shape => [batch, 1, conv_units]
    #     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=CONV_WIDTH),
    #     # Shape => [batch, 1,  out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features,
    #                           kernel_initializer=tf.initializers.zeros),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])

    # history = compile_and_fit(multi_conv_model, multi_window)
    #
    # val_p['Conv'] = multi_conv_model.evaluate(multi_window.val)
    # p['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
    # multi_window.plot(plot_col=info['obj_var'][0], model=multi_conv_model)
    #
    #
    # # Refractor into forecasting function inside the cnn class
    # test = multi_window.make_dataset(dt_test.loc[0:57, :])
    # inputs, labels = next(iter(test))
    # tmp = multi_conv_model.predict(inputs)
    #
    # from matplotlib.pyplot import plot as plt
    # from matplotlib.pyplot import cla
    #
    # cla()
    # plt(range(7), inputs[0, :, 34]) # Initial 7 values provided
    # plt(range(6, 56), labels[0, :, 0]) # Expected 50 values afterwards
    # #plt(range(6, 56), dt_test[info['obj_var']][8:58]) # Real 50 values
    # plt(range(6, 56), tmp[0, :, 0]) # Predicted 50 values

