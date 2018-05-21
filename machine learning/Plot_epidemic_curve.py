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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import multiprocessing
import time
import random as rand
import plotly.plotly as py
import plotly.graph_objs as go
from joblib import Parallel, delayed
import json
from sklearn.metrics import confusion_matrix



# This is read from the h5 file
FEATURE_NAMES = ['unique_locations_visited','calls_count_outgoing','mean_call_duration_both']
FEATURE_COUNT = 0

NUMBER_OF_CPU_CORES = multiprocessing.cpu_count()
PARALLEL = Parallel(n_jobs=NUMBER_OF_CPU_CORES)


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
            '../../inputData/learning_data/%s_hour_bins/%s/%s/3_features_%d_days.h5' 
                % (hours, time, file, 29 if time == "relative" else 250 ) 
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

def get_epidemic_curve(model, input_sets, offsets):
    date_predicted = {}
    predicted = model.predict(input_sets)
#    predicted = np.reshape(predicted,(-1,1))
    for off, pre in zip(offsets, predicted):
        date_predicted.setdefault(str(off), []).extend(pre.tolist())
    return date_predicted

def get_epidemic_curve_labels(label_sets, offsets):
    date_label = {}
    for off, label in zip(offsets, label_sets):
            date_label.setdefault(str(off), []).extend(list(label))
    return date_label

def find_threshold(offset_pred_dict, actual_curve):
    print("""
    """)
    best_threshold = 0
    best_sum = 100000000 
    actual = np.array([list(v) for k,v in actual_curve.items()])
    actual = np.sum(actual, axis=1)
    nparr = np.array([list(v) for k,v in offset_pred_dict.items()])
    for threshold in np.arange(0.0, 1, 0.0005):
        cur_sum = np.count_nonzero(nparr > threshold, axis=1) 
        cur_sum = np.absolute(cur_sum - actual)
        cur_sum = sum(cur_sum)
        if best_sum >= cur_sum:
            best_threshold = threshold
            best_sum = cur_sum
            #print("best threshold: %f %f" % (best_threshold, cur_sum))
    return best_threshold

def plot_curves(actual_curve, curves, days):
    actual_train, _, _ = actual_curve
    traces = []
    for train_pre, test_pre, name in curves:
        if actual_train != train_pre:
            threshold = find_threshold(train_pre, actual_train)
        else:
            threshold = .5
        print(name, threshold)
        x, y = [], []
        for off, pred in test_pre.items():
            # dt.date(2009, 6, 4)
            # dt.date(2009, 8, 4)
            x.append(dt.date(2009, 6, 4) + dt.timedelta(int(off)))
            y.append(sum(i > threshold for i in pred))
        
        traces.append(go.Scatter(
            x=x,
            y=y,
            name=name,
            mode='lines',
            line=dict(
                shape='spline'
            )
        ))
    py.sign_in('thorgeirk11', '??')  # Pro features
        
    layout = {
        "autosize": True, 
        "width": 700, 
        "height": 500,
        "title": "Epidemic Curve<br>Models trained on top 3 features and 30 days<br> Labeled sick %s" % ("on the last day" if days == 1 else "in the last 7 days"), 
        "xaxis": {
            "autorange": True, 
            "range": ["2009-07-04", "2009-10-01"], 
            "title": "90 day period during the H1N1 outbreak", 
            "type": "date"
        }, 
        "yaxis": {
            "autorange": True, 
            "dtick": 25, 
            "range": [1.05555555556, 199.944444444], 
            "tickmode": "linear", 
            "title": "Individuals diagnosed", 
            "type": "linear"
        }
    }
    fig = go.Figure(data=traces, layout=layout)
    #py.image.save_as(fig, 'epidemic_curves')
    print(py.plot(fig, filename='epidemic_curves4_top_last_%d' % days))

def plot_pr_auc(actual_curve, curves, days):
    actual_train, actual_test, _ = actual_curve
    actual_test = [v for vs in actual_test.values() for v in vs]
    traces = []
    for train_pre, test_pre, name in curves:
        if actual_train == train_pre:
            continue  
        test_pre = [v for vs in test_pre.values() for v in vs]
        average_precision = average_precision_score(actual_test, test_pre)
        precision, recall, _ = precision_recall_curve(actual_test, test_pre)
        traces.append(go.Scatter(
            x=recall,
            y=precision,
            name='%s (%0.2f AUC)' % (name, average_precision),
            mode='lines',
            line=dict(
                shape='spline'
            )
        ))          
    py.sign_in('thorgeirk11', '??')  # Pro features
        
    layout = {
        "autosize": True, 
        "width": 500, 
        "height": 500,
        "title": "Precision Recall Curves<br>Models trained on top 3 features and 30 days<br> Labeled sick %s" % ("on the last day" if days == 1 else "in the last 7 days"), 
        "xaxis": {
            "autorange": True, 
            "title": "Recall", 
        }, 
        "yaxis": {
            "autorange": True, 
            "title": "Precision", 
        }
    }
    fig = go.Figure(data=traces, layout=layout)
    #py.image.save_as(fig, 'epidemic_curves')
    print(py.plot(fig, filename='precision_recall_curves2_%d' % days))


def write_scores(model, name, inputs, offsets, types):
    for model_input, offset, type in zip(inputs, offsets, types):
        if name == 'actual_curve':
            curve = get_epidemic_curve_labels(model_input, offset)
        else:
            curve = get_epidemic_curve(model, model_input, offset)
        path = 'Predictions_250/%s' % (type)
        if not os.path.exists(path):
            os.makedirs(path)
        with open('Predictions_250/%s/%s.json' % (type, name), 'w') as fp:
            json.dump(curve, fp)


if False:
    is_test = False
    delay = 7

    prod_models = { 
        1: {
            'dense_model': 
                "results_250days/absolute/3_features_4h_8h_24h_48h_72h_96h/test_0/30days/fully_connected_AUC_0.699_Precision_0.086.hdf5",
            'gru_model': 
                "results_250days/absolute/3_features_4h_8h_24h_48h_72h_96h/test_1/30days/gru_AUC_0.709_Precision_0.090.hdf5",
            'bidir_gru_model': 
                "results_250days/absolute/3_features_4h_8h_24h_48h_72h_96h/test_1/30days/bidir_gru_AUC_0.737_Precision_0.093.hdf5",
        },        
        #1: {
        #    'dense_model': 
        #        "results/absolute/3_features_4h_8h_24h_48h_72h_96h/test_20/30days/fully_connected_AUC_0.668_Precision_0.116.hdf5",
        #    'gru_model': 
        #        "results/absolute/3_features_4h_8h_24h_48h_72h_96h/test_20/30days/gru_AUC_0.693_Precision_0.148.hdf5",
        #    'bidir_gru_model': 
        #        "results/absolute/3_features_4h_8h_24h_48h_72h_96h/test_20/30days/bidir_gru_AUC_0.697_Precision_0.154.hdf5",
        #},
        7:{
            'dense_model': 
                "results_250days_7d_delay/absolute/3_features_4h_8h_24h_48h_72h_96h/test_11/30days/fully_connected_AUC_0.698_Precision_0.277.hdf5",
            'gru_model': 
                "results_250days_7d_delay/absolute/3_features_4h_8h_24h_48h_72h_96h/test_1/30days/gru_AUC_0.714_Precision_0.301.hdf5",
            'bidir_gru_model': 
                "results_250days_7d_delay/absolute/3_features_4h_8h_24h_48h_72h_96h/test_0/30days/bidir_gru_AUC_0.749_Precision_0.200.hdf5",
        }
    }

    test_models_dict = {
        'dense_model': 
            "results_new/absolute/1_features_24h/test_0/5days/fully_connected_AUC_0.653_Precision_0.109.hdf5",
        'gru_model': 
            "results_new/absolute/1_features_24h/test_0/5days/gru_AUC_0.654_Precision_0.109.hdf5",
        'bidir_gru_model': 
            "results_new/absolute/1_features_24h/test_0/5days/bidir_gru_AUC_0.654_Precision_0.109.hdf5",
    }
    test_models = { 1: test_models_dict, 7: test_models_dict}



    TRUE_LABEL_RANGE = range(-1,4 + delay)
    print(TRUE_LABEL_RANGE)
    days = "%dday" % delay

    num_top_features = 1 if is_test else 3
    hour_per_bin = [24] if is_test else [4,8,24,48,72,96]
    sub_seq_count = 5 if is_test else 30
    data_type = ["absolute"]

    in_train2d, out_train, offset_train = read_multiple(hour_per_bin, 'sick_train', data_type, sub_seq_count, num_top_features, True)
    in_test2d, out_test, offset_test = read_multiple(hour_per_bin, "sick_test", data_type, sub_seq_count, num_top_features, True)

    in_train2d = np.concatenate(in_train2d, 1)
    in_test2d = np.concatenate(in_test2d, 1)

    print("\n-----------")
    print(in_train2d.shape)
    print(in_test2d.shape)
    print("-----------")
    input_shape_2d = (in_train2d.shape[1], )

    write_scores(None, 'actual_curve', 
        [out_train, out_test], 
        [offset_train, offset_test], 
        ['%s/train' % days,'%s/test' % days]
    ) 

    # =============================================
    #                Linear Model
    # =============================================

    model = LinearRegression()
    model.fit(in_train2d, out_train)

    write_scores(model, 'linear_curve', 
        [in_train2d, in_test2d], 
        [offset_train, offset_test], 
        ['%s/train' % days,'%s/test' % days]
    ) 

    # =============================================
    #            Fully Connected Network
    # =============================================

    #in_layer = Input(input_shape_2d, name=FEATURE_NAMES[0])
    #layer = Dense(32, activation="relu")(in_layer)
    #layer = Dropout(0.5)(layer)
    #layer = Dense(16, activation="relu")(layer)
    #output_layer = Dense(1, activation="sigmoid")(layer)

    #model = Model(inputs=in_layer, outputs=output_layer)
    #model.compile(
    #    optimizer='adam',
    #    loss='binary_crossentropy'
    #)
    #model.load_weights((test_models if is_test else prod_models)[delay]["dense_model"])

    #write_scores(model, 'dense_curve', 
    #    [in_train2d, in_test2d], 
    #    [offset_train, offset_test], 
    #    ['%s/train' % days,'%s/test' % days]
    #) 
    ## =============================================
    ##                   RNN
    ## =============================================
    #del in_train2d
    #del in_test2d

    in_train, out_train, _ = read_multiple(hour_per_bin, 'sick_train', data_type, sub_seq_count, num_top_features, False)
    in_test, out_test, _ = read_multiple(hour_per_bin, "sick_test", data_type, sub_seq_count, num_top_features, False)


    input_shape = (in_train.shape[2], in_train.shape[3])
    #gru_model = create_model(input_shape, num_top_features, CuDNNGRU(10))
    #gru_model.compile(
    #    optimizer='adam',
    #    loss='binary_crossentropy'
    #)
    #gru_model.load_weights((test_models if is_test else prod_models)[delay]["gru_model"])
    #write_scores(gru_model, 'gru_curve', 
    #    [list(in_train), list(in_test)], 
    #    [offset_train, offset_test], 
    #    ['%s/train' % days,'%s/test' % days]
    #) 

    bidir_gru_model = create_model(input_shape, num_top_features, Bidirectional(CuDNNGRU(10)))
    bidir_gru_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )
    bidir_gru_model.load_weights((test_models if is_test else prod_models)[delay]["bidir_gru_model"])
    write_scores(bidir_gru_model, 'bidir_gru_curve', 
        [list(in_train), list(in_test)], 
        [offset_train, offset_test], 
        ['%s/train' % days,'%s/test' % days]
    ) 
else:
    for days in [1,7]:
        curves_names =  [
            ('actual_curve', "Actual"),
            ('linear_curve', "Linear"),
            #('logistic_curve', "Logistic"),
            #('dense_curve', "Dense"),
            #('gru_curve', "GRU"),
            ('bidir_gru_curve', "Bidir_GRU")
        ]

        curves = []
        for path, name in curves_names:
            with open('Predictions_250/%dday/test/%s.json' % (days,path), 'r') as test:
                with open('Predictions_250/%dday/train/%s.json' % (days,path), 'r') as train:
                    curves.append((json.load(train), json.load(test), name))

        #plot_curves(curves[0], curves, days)
        plot_pr_auc(curves[0], curves, days)
