import tensorflow as tf
from pandas import DataFrame
from pandas import concat
import numpy as np
from collections import Counter
import pandas as pd
import h5py
from tqdm import tqdm
from keras.layers import *
from keras.models import Model, Sequential
from keras.callbacks import *
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
from keras.utils import plot_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve,  average_precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import random as rand

# This is read from the h5 file
FEATURE_NAMES = []
FEATURE_COUNT = 0
#
#HALF_LABEL_RANGE = range(-2,4)


# =============================================
#                   Utils
# =============================================

def get_class_weights(y):
    num_of_one = np.sum(y)
    size = y.size
    delta = {
        0: 1 - num_of_one / size,
        1: num_of_one / size
    }
    weights = {
        0: 1,
        1: (size - num_of_one) / num_of_one 
    }
    print(delta, weights)
    return weights

def normalize_to_minus_one_to_one(data):
    return ((np.array(data) - 0.5) * 2).tolist()

# =============================================
#                Retrive Data
# =============================================

def get_time_series(user_data, labels, feature_index, num_bins, num_features, slice_size):
    try:
        true_i = 0
        true_i = labels.index(1)
        for index in [true_i + i for i in TRUE_LABEL_RANGE]:
            try:
                labels[index] = 1.0
            except:
                pass
    except:
        pass  
    
    feature_data = user_data[feature_index::num_features]
    num_days = feature_data.shape[0]
    total = num_days - slice_size * num_bins
    for start in range(0, total, num_bins):
        end = start + slice_size * num_bins
        label_end = start // num_bins + slice_size
        yield (
            # each time step is equal to the number of bins for that day
            normalize_to_minus_one_to_one([
                feature_data[i:i + num_bins] 
                for i in range(start, end, num_bins)
            ]),
            #Find the label for the day at the end of the peroid.
            [labels[label_end]],
            label_end,
        )
def setup_labels_and_sliding_window(data, num_features, sub_seq_count, top_feature_count, flat_array):
    all_features = []
    all_labels = []
    all_offset = []
    for feature_index in tqdm(range(top_feature_count)):
        user_series = []
        for inputs, labels, num_bins in data:
            data_series, all_labels, all_offset = [], [], []
            for i in range(len(inputs)):
                time_series = get_time_series(
                    inputs[i], 
                    list(labels[i]), 
                    feature_index, 
                    num_bins, 
                    num_features, 
                    sub_seq_count
                )
                for s, l, o in time_series:
                    if flat_array:
                        data_series.append(np.array(s).flatten())
                    else:
                        data_series.append(s)
                    all_labels.append(l)
                    all_offset.append(o)
            if len(user_series) == 0:
                user_series.extend(data_series)
            else:
                #print(np.array(user_series).shape)
                #print(np.array(data_series).shape)
                user_series = np.concatenate((user_series, data_series), 1 if flat_array else 2)
                #print("CONCATING TOGETHER!")
        all_features.append(user_series)    
    x, y = np.array(all_features), np.array(all_labels)
    return x, y, np.array(all_offset)

def read_data(files, sub_seq_count, top_feature_count,  flat_array):
    global FEATURE_NAMES, FEATURE_COUNT
    data = []
    for file in files:
        print("Reading: %s" % file)
        with h5py.File(file, 'r') as datafile:
            inputs = datafile['inputs'][:]
            labels = datafile['labels'][:]
            FEATURE_NAMES = [str(name, "utf-8") for name in datafile['feature_names'][:]]
            #print(FEATURE_NAMES)
            FEATURE_COUNT = len(FEATURE_NAMES)
            number_of_bins = datafile['bin_count'][0]
            data.append((inputs, labels, number_of_bins))
    return setup_labels_and_sliding_window(data, FEATURE_COUNT, sub_seq_count, top_feature_count, flat_array)

def read_multiple(hour_bins, file, times, sub_seq_count, top_feature_count=0, flat_array=False):
    ins, outs, offsets = [], [], []
    for time in times:
        files = [
            '../../inputData/learning_data/%s_hour_bins/%s/sick_%s/3_features_%d_days.h5' 
                % (hours, time, file, 29 if time == "relative" else 250) 
            for hours in hour_bins
        ]
        data, labels, off = read_data(
            files,
            sub_seq_count,
            top_feature_count,
            flat_array
        )
        if len(ins) == 0:
            ins.extend(data)
        else:
            print(np.array(ins).shape)
            print(np.array(data).shape)
            ins = np.concatenate((ins,data), axis=1)
            print(np.array(ins).shape)
        outs.extend(labels)
        offsets.extend(off)
    return np.array(ins), np.array(outs), np.array(offsets)


class EarlyStoppingOnAUC(TensorBoard):
    def __init__(self, patience, in_test_data, log_dir, best_filepath):
        super(EarlyStoppingOnAUC, self).__init__(log_dir)
        self.best_precision = 0
        self.wait = 0
        self.in_test_data = in_test_data
        self.patience = patience
        self.best_filepath = best_filepath
 
    def on_epoch_end(self, epoch, logs=None):
        out_pred = self.model.predict(self.in_test_data).flatten()
        fpr, tpr, _ = roc_curve(out_test.flatten(), out_pred)
        cur_auc = auc(fpr, tpr)
        logs["val_auc"] = cur_auc
        
        average_precision = average_precision_score(out_test.flatten(), out_pred)
        logs["val_avg_prec"] = average_precision
        
        #print(cur_auc)
        self.wait += 1
        if self.best_precision < average_precision :
            self.model.save_weights(self.best_filepath, overwrite=True)
            self.best_precision = average_precision
            self.time_best = time.time()
            self.wait = 0
        if self.wait >= self.patience and time.time() > self.time_best + 60:
            self.model.stop_training = True
        super(EarlyStoppingOnAUC, self).on_epoch_end(epoch, logs)

def write_csv(data, name):  
    np.savetxt("%s.csv" % name, data, fmt="%.15g", delimiter=",")

def create_model(input_shape, feature_count, center_layer):
    feature_layers = []
    inputs = []
    if feature_count > 1:
        for feature in range(feature_count):
            layer = Input(input_shape, name=FEATURE_NAMES[feature])
            inputs.append(layer)
            feature_layers.append(layer)
        layer = concatenate(feature_layers)
    else:
        layer = Input(input_shape, name=FEATURE_NAMES[0])
        inputs.append(layer)

    layer = center_layer(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(16, activation="relu")(layer)
    output_layer = Dense(1, activation="sigmoid")(layer)

    return Model(inputs=inputs, outputs=output_layer)


def plot_auc(roc_auc, fpr, tpr, title, path):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("%s\nROC curve (area = %0.3f)" % (title, roc_auc))
    plt.savefig("%s_AUC_%0.3f.pdf" % (path, roc_auc))
def plot_pre_rec(recall, precision, average_precision, title, path):
    plt.figure()        
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(
        '%s\nPrecision-Recall curve: AP=%0.3f' % 
        (title, average_precision)
    )
    plt.savefig("%s_Precision_%0.3f.pdf" % (path, average_precision))

def generate_results(y_test, y_score, num_top_features, path):
    print(sum(y_test), y_test[:40])
    print(sum(y_score), y_score[:40])

    title = '%sh bins and %d days on %s' % (
        "h, ".join([str(h)for h in hour_per_bin]),
        sub_seq_count,
        "Unique Locations Visited" if num_top_features == 1 else ("%d features" % num_top_features)
    )    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plot_auc(roc_auc, fpr, tpr, title, path)

    average_precision = average_precision_score(y_test, y_score)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    plot_pre_rec(recall, precision, average_precision, title, path)

    return "AUC_%0.3f_Precision_%0.3f" % (roc_auc,average_precision), roc_auc, average_precision

    
def fit_and_plot_auc(model_name, in_train_data, in_test_data, num_top_features, path):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())

        model, model_type = model_name
        model.summary()
        best_filepath = "%s_current_best.hdf5" % path 
        tensorboard_dir = './logs/%s%s' % (path[8:], model_type)
        early_stop = EarlyStoppingOnAUC(
            patience = 20, 
            in_test_data = in_test_data, 
            log_dir = tensorboard_dir,
            best_filepath =best_filepath
        )
        early_stop.set_model(model)

        model.fit(
            x=in_train_data,
            y=out_train,
            batch_size=128,
            epochs=10000,
            validation_data=(
                in_test_data,
                out_test
            ),
            verbose=1,
            class_weight=get_class_weights(out_train),
            callbacks=[early_stop]
        )
        model.load_weights(best_filepath)
        test_predict = model.predict(in_test_data).flatten()

        test_true = np.array(out_test).flatten()
        path = "%s%s" % (path, model_type)
        name , roc_auc, average_precision = generate_results(test_true, test_predict, num_top_features, path)
        name = "%s_%s" % (path, name)
        model.save_weights("%s.hdf5" % name)
        plot_model(model, to_file="%s.pdf" % name, show_shapes=True, show_layer_names=True, rankdir='TB')
        with open("%s.txt" % name, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        return roc_auc, average_precision 



REPEAT_COUNT = range(3,10)
TRUE_LABEL_RANGE = range(-1,5) #range(-1,12)
NUM_TOP_FEATURES = [3]
HOURS_PER_BIN_LIST = [[4,8,24,48,72,96]]
SUB_SEQUENCE_COUNTS = range(30, 31, 5)
DATA_TO_READ = [["absolute"]]
for data_type in DATA_TO_READ:
    for num_top_features in NUM_TOP_FEATURES:
        for hour_per_bin in HOURS_PER_BIN_LIST:
 
            in_train, out_train, _ = read_multiple(hour_per_bin, 'train', data_type, 30, num_top_features, False)
            in_test, out_test, _ = read_multiple(hour_per_bin, "test", data_type, 30, num_top_features, False)
            print("\n-----------")
            print(in_train.shape)
            print(out_train.shape)
            print("-----------")
            print(in_test.shape)
            print(out_test.shape)
            print("-----------")

            for repeat_test in REPEAT_COUNT:
                root_path = "results_250days/%s/%d_features_%sh/test_%d/"% (
                    "_and_".join(data_type),
                    num_top_features,
                    "h_".join([str(h) for h in hour_per_bin]),
                    repeat_test
                )
                write_header = False
                if not os.path.exists(root_path):
                    os.makedirs(root_path)
                    write_header =True
                with open("%stable%d.tex" % (root_path, repeat_test), "a") as tex_table:
                    if write_header:
                        tex_table.write("""\\begin{tabular}{|c||c|c|c|c|c|c|c|c|}\\hline
                & \\multicolumn{2}{c|}{Linear} & \\multicolumn{2}{c|}{Dense} & \\multicolumn{2}{c|}{GRU} & \\multicolumn{2}{c|}{Bid-GRU} \\\\\\cline{1-9}
            Days & AUC & Pre & AUC & Pre & AUC & Pre & AUC & Pre \\\\\\hline\n""")
                    path = "%s%ddays/" % (root_path, 30)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    else:
                        continue
                    print("THIS IS THE FILE PATH NOW!")
                    print(path)
                    print("==========================")
                    #tex_table.write("    %d " % sub_seq_count)
                    #tex_table.flush()

                    #in_train2d, out_train, _ = read_multiple(hour_per_bin, 'train', data_type, sub_seq_count, num_top_features, #True)
                    #in_test2d, out_test, _ = read_multiple(hour_per_bin, "test", data_type, sub_seq_count, num_top_features, True)
                    #in_train2d = np.concatenate(in_train2d, 1)
                    #in_test2d = np.concatenate(in_test2d, 1)

                    #print("\n-----------")
                    #print(in_train2d.shape)
                    #print(in_test2d.shape)
                    #print("-----------")
                    #input_shape_2d = (in_train2d.shape[1], )


                    ## =============================================
                    ##                Linear Model
                    ## =============================================

                    #model = LinearRegression()
                    #model.fit(in_train2d, out_train)
                    #test_predict = model.predict(in_test2d).flatten()
                    #test_true = np.array(out_test).flatten()
                    #linearpath = "%s%s" % (path, 'linear')
                    #name , roc_auc, average_precision = generate_results(test_true, test_predict, num_top_features, linearpath)
                    #name = "%s_%s" % (linearpath, name)
                    #tex_table.write("& %0.3f & %0.3f " % (roc_auc, average_precision))
                    ## =============================================
                    ##            Fully Connected Network
                    ## =============================================

                    #in_layer = Input(input_shape_2d, name=FEATURE_NAMES[0])
                    #layer = Dense(32, activation="relu")(in_layer)
                    #layer = Dropout(0.5)(layer)
                    #layer = Dense(16, activation="relu")(layer)
                    #output_layer = Dense(1, activation="sigmoid")(layer)

                    #fully_connected = Model(inputs=in_layer, outputs=output_layer)
                    #fully_connected.compile(
                    #    optimizer='adam',
                    #    loss='binary_crossentropy'
                    #)
                    #roc_auc, average_precision = fit_and_plot_auc(
                    #    (fully_connected, "fully_connected"), 
                    #    in_train2d,
                    #    in_test2d,
                    #    num_top_features,
                    #    path
                    #)
                    #tex_table.write("& %0.3f & %0.3f" % (roc_auc, average_precision))

                    ## =============================================
                    ##                   RNN
                    ## =============================================
                    #del in_train2d
                    #del in_test2d
                    input_shape = (in_train.shape[2], in_train.shape[3])
                    #gru_model = create_model(input_shape, num_top_features, CuDNNGRU(10))
                    bidir_gru_model = create_model(input_shape, num_top_features, Bidirectional(CuDNNGRU(10)))

                    for m, name in [(bidir_gru_model, "bidir_gru")]:
                        m.compile(
                            optimizer='adam',
                            loss='binary_crossentropy'
                        )
                        roc_auc, average_precision = fit_and_plot_auc(
                            (m, name), 
                            list(in_train), 
                            list(in_test), 
                            num_top_features,
                            path
                        )
                        tex_table.write(" & %0.3f & %0.3f" % (roc_auc, average_precision))
                    tex_table.write("\\\\\n")
                tex_table.write("\\hline\n\\end{tabular}")
