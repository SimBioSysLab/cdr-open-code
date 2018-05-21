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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import random as rand

#FEATURE_NAMES = []
FEATURE_COUNT = 3
SUB_SEQUENCE_COUNT = 8
OFFSET_RANGE = range(0,9)


def read_data(file, random_seed, test_slice=None):
    #global FEATURE_NAMES, FEATURE_COUNT
    with h5py.File(file, 'r') as datafile:
        inputs = datafile['inputs'][:]
        labels = datafile['labels'][:]
        #offset = datafile["offsets"][:]
        #FEATURE_NAMES = [str(name, "utf-8") for name in datafile['feature_names'][:]]
        #FEATURE_COUNT = len(FEATURE_NAMES)
        return inputs, labels #, setup_labels_and_sliding_window(inputs, offset, labels, random_seed, test_slice)

X, Y = read_data('learning_data/top3/sick/sick_train_2_day.h5', 1337)
X_test, Y_test = read_data('learning_data/top3/sick/sick_test_2_day.h5', 42)

def generate_results(y_test, y_score, name):
    print(np.array(y_test[:40]).flatten())
    print(np.array(y_score[:40]).flatten())
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s\nROC curve (area = %0.3f)' % (name,roc_auc))
    #plt.savefig('AUC_%f_%s.pdf' % (roc_auc,name))
    #plt.show()
    print('ROC_AUC: %f %s' % (roc_auc,name))

gnb = GaussianNB()
gnb.fit(X,Y)
pred = gnb.predict(X_test)
generate_results(Y_test, pred, 'GaussianNB')

baysian = linear_model.BayesianRidge()
baysian.fit(X,Y)
pred = baysian.predict(X_test)
generate_results(Y_test, pred, 'BayesianRidge')

forset = RandomForestClassifier()
forset.fit(X,Y)
pred = forset.predict(X_test)
generate_results(Y_test, pred, 'RandomForestClassifier')

linear = Sequential([
    Dense(21, input_dim=21, kernel_initializer='normal', activation="relu"),
    Dense(1, kernel_initializer='normal')
])
linear.compile(optimizer='adam', loss='mse', metrics=["acc"])
linear.fit(X,Y,verbose=0)
pred = linear.predict(X_test)
generate_results(Y_test, pred, 'linear_regression keras')


#small = Sequential([
#    Conv1D(8, 3, strides=3, activation='relu', input_shape=(None, 21)),
#    MaxPooling1D(1),
#    Dense(1, activation="sigmoid", kernel_initializer='normal')
#])
#small.compile(optimizer='adam', loss='mse', metrics=["acc"])
#small.summary()
#small.fit(X,Y,verbose=0)
#pred = small.predict(X_test)
#generate_results(Y_test, pred, 'small_dense keras')

