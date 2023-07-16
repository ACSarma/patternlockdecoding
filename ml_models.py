import datetime
import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPool1D
from keras.models import Sequential
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from keras.optimizers import Adam


def create_lstm_model(lr, dr, resolution):
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(resolution, 4)))
    model.add(Dropout(dr))
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(dr))
    # model.add(Dropout(dr))
    # model.add(LSTM(units=128, return_sequences=True))
    # model.add(Dropout(dr))
    # model.add(LSTM(units=64, return_sequences=True))
    # model.add(Dropout(dr))
    # model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(units=6, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def create_cnn_model(resolution):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=7, input_shape=(resolution, 4)))  # 1
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=10))  # 2
    model.add(Dropout(0.4))
    model.add(Conv1D(filters=64, kernel_size=3))  # 3
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=10))  # 4
    model.add(Conv1D(filters=32, kernel_size=2))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))  # 11

    opt = Adam(learning_rate=0.0001)
    model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
