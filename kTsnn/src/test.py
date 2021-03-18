from kTsnn.src.utils import *
from kTsnn.src.nets.cnn import Conv1dnn
from kTsnn.src.nets.window_gen import WindowGenerator

MAX_EPOCHS = 40
DT_FILE = 'dt_unfolded.csv'
INFO_FILE = 'exec_info.txt'

if __name__ == '__main__':
    dt = load_dt(DT_FILE)
    info = load_info(INFO_FILE)
    batch = 10
    num_features = dt.shape[1]

    dt_train, dt_test, dt_val = get_train_test_val(dt, info['test'], info['val'], info['idx_cyc'])
    #dt_train, dt_test, dt_val = norm_dt(dt_train, dt_test, dt_val)

    # Forecasting into the future
    OUT_STEPS = 50
    INPUT_WIDTH = 7
    multi_window = WindowGenerator(input_width=INPUT_WIDTH, label_width=OUT_STEPS, shift=OUT_STEPS,
                         dt_train=dt_train, dt_test=dt_test, dt_val=dt_val,
                         label_columns=info['obj_var'])
    CONV_WIDTH = 7
    ini = 0

    model = Conv1dnn(MAX_EPOCHS, multi_window, num_features, conv_width=CONV_WIDTH, out_steps=OUT_STEPS)
    model.train_net()
    model.fit_net(patience=None)
    pred = model.predict(dt_test.loc[ini:(ini+INPUT_WIDTH+OUT_STEPS), :], obj_var=info['obj_var'][0])



