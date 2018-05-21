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


#inputs = np.array([  
#    [3,2,5],
#    [4,7,8],
#    [9,1,3]
#])
#labels = np.array([
#    [1], [0], [1]
#])
#
#
#lreg = LinearRegression(fit_intercept=False)
#lreg.fit(inputs, labels)
#w = lreg.coef_
#print(w)
#print(np.matmul(inputs, w[0]))
#exit()
#
#
#model = Sequential([
#     Dense(1, activation="linear", input_shape=(3,))
#])
#model.compile(
#    optimizer="adam",
#    loss='binary_crossentropy'
#)
#
#model.fit(inputs,labels, epochs=10)
#print(model.get_weights())





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

def get_time_series(user_data, labels, feature_index, num_bins, num_features, slice_size,test):
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
            [max([
                    labels[label_end-offset] 
                    for offset in 
                    (LABEL_DAY_OFFSET_TEST if test else LABEL_DAY_OFFSET_TRAIN)
                ])],
            label_end,
        )
def setup_labels_and_sliding_window(data, num_features, sub_seq_count, top_feature_count, test, flat_array):
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
                    sub_seq_count,
                    test
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

def read_data(files, sub_seq_count, top_feature_count, test, flat_array):
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
    return setup_labels_and_sliding_window(data, FEATURE_COUNT, sub_seq_count, top_feature_count, test, flat_array)

def read_multiple(hour_bins, file, times, sub_seq_count, top_feature_count, days, flat_array=False):
    ins, outs, offsets = [], [], []
    for time in times:
        files = [
            '../../inputData/learning_data/%s_hour_bins/%s/sick_%s/43_features_%d_days.h5' 
                % (hours, time, file, days) 
            for hours in hour_bins
        ]
        data, labels, off = read_data(
            files,
            sub_seq_count,
            top_feature_count,
            file == "test",
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
        out_pred = self.model.predict(self.in_test_data, batch_size=2048).flatten()
        fpr, tpr, _ = roc_curve(out_test.flatten(), out_pred)
        cur_auc = auc(fpr, tpr)
        logs["val_auc"] = cur_auc
        
        average_precision = average_precision_score(out_test.flatten(), out_pred)
        logs["val_avg_prec"] = average_precision
        
        print(cur_auc, average_precision)
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

def create_deep(input_shape, feature_count, center_layer):
    feature_layers = []
    inputs = []
    if feature_count > 1:
        for feature in range(feature_count):
            layer = Input(input_shape, name=FEATURE_NAMES[feature])
            inputs.append(layer)
            layer = Bidirectional(CuDNNGRU(units=10,return_sequences=True))(layer)
            feature_layers.append(layer)
        layer = concatenate(feature_layers)
    else:
        layer = Input(input_shape, name=FEATURE_NAMES[0])
        layer = center_layer(layer)
        inputs.append(layer)

    layer = Bidirectional(CuDNNGRU(units=4,return_sequences=True))(layer)
    layer = Bidirectional(CuDNNGRU(units=3,return_sequences=True))(layer)
    layer = Bidirectional(CuDNNGRU(units=2,return_sequences=True))(layer)
    layer = CuDNNGRU(units=1)(layer)
    #layer = Dense(5, activation="relu")(layer)
    #layer = Dense(10, activation="relu")(layer)
    #layer = Dense(5, activation="relu")(layer)
    #layer = Dense(1, activation="sigmoid")(layer)

    return Model(inputs=inputs, outputs=layer)

def create_shallow(input_shape, feature_count, center_layer):
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
        layer = center_layer(layer)
        inputs.append(layer)

    layer = Bidirectional(CuDNNGRU(units=10))(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(8, activation="relu")(layer)
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
    print(y_test[:40])
    print(y_score[:40])
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
            patience = 60, 
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
        train_predict = model.predict(in_train_data).flatten()
        test_predict = model.predict(in_test_data).flatten()
        test_true = np.array(out_test).flatten()
        path = "%s%s" % (path, model_type)
        name , roc_auc, average_precision = generate_results(test_true, test_predict, num_top_features, path)
        name = "%s_%s" % (path, name)
        model.save_weights("%s.hdf5" % name)
        plot_model(model, to_file="%s.pdf" % name, show_shapes=True, show_layer_names=True, rankdir='TB')
        with open("%s.txt" % name, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        return roc_auc, average_precision, train_predict, test_predict


TRUE_LABEL_RANGE = range(-1,5)
LABEL_DAY_OFFSET_TRAIN = range(1)
LABEL_DAY_OFFSET_TEST = range(1)
num_top_features = 3
hour_per_bin = [24]
sub_seq_count = 30
data_type = "absolute"

root_path = "experiment_results/%s/%d_features_%sh_offset_%d_%d/"% (
    data_type,
    num_top_features,
    "h_".join([str(h) for h in hour_per_bin]),
    max(LABEL_DAY_OFFSET_TRAIN),
    max(LABEL_DAY_OFFSET_TEST)
)

path = "%s%ddays_rand%d/" % (root_path, sub_seq_count, rand.randint(1,1000))
if not os.path.exists(path):
    os.makedirs(path)
print("THIS IS THE FILE PATH NOW!")
print(path)
print("==========================")

cached_data_file = "%sexported_data.h5" % root_path
in_train, out_train, train_offsets = read_multiple(hour_per_bin, 'train', [data_type], sub_seq_count, num_top_features, 120, False)
in_test, out_test, test_offsets = read_multiple(hour_per_bin, 'test', [data_type], sub_seq_count, num_top_features, 120, False)

print("\n-----------")
print(in_train.shape)
print(out_train.shape)
print("-----------")
print(in_test.shape)
print(out_test.shape)
print("-----------")

input_shape = (in_train.shape[2], in_train.shape[3])


rnn_layer = Bidirectional(CuDNNGRU(10))
feature_layers = []
inputs = []
for feature in range(num_top_features):
    layer = Input(input_shape, name=FEATURE_NAMES[feature])
    inputs.append(layer)
    layer2 = rnn_layer(layer)
    feature_layers.append(layer2)
    layer = Bidirectional(CuDNNGRU(5))(layer)
    feature_layers.append(layer)
    
layer = concatenate(feature_layers)
layer = Dropout(0.5)(layer)
layer = Dense(16, activation="relu")(layer)
output_layer = Dense(1, activation="sigmoid")(layer)

bidir_gru_model = Model(inputs=inputs, outputs=output_layer)
bidir_gru_model.compile(
    optimizer='adam', 
    loss='binary_crossentropy'
)
roc_auc, average_precision, train_predict, test_predict  = fit_and_plot_auc(
    (bidir_gru_model, "bidir_gru_fun5"), 
    list(in_train), 
    list(in_test), 
    num_top_features,
    path
)
print(roc_auc, average_precision)

# lr = LinearRegression()
# lr.fit()
# 
# x=in_train_data,
# y=out_train,
# batch_size=2048,
# epochs=10000,
# validation_data=(
#     in_test_data,
#     out_test
# ),
# verbose=1,
# class_weight=get_class_weights(out_train),