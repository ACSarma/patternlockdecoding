'''
Author: Arun Sarma
Course: CSS595
Purpose: This script has functions to create models based on hyperparameters and data resolution, and returns them.
This is used by model_evaluation.py
'''

from keras.layers import Conv1D
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers import Flatten
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import MaxPool1D
from keras.models import Sequential
from keras.optimizers import Adam


def create_lstm_model(lr, dr, num_labels, num_hidden, in_shape):
    curr_units = 2048

    model = Sequential()
    model.add(LSTM(units=curr_units, return_sequences=True, input_shape=in_shape))

    for layer in range(num_hidden):
        curr_units = curr_units / 2
        model.add(LSTM(units=int(curr_units), return_sequences=True))
        model.add(Dropout(dr))

    curr_units = curr_units / 2
    model.add(LSTM(units=int(curr_units), return_sequences=False))
    model.add(Dense(units=36))
    model.add(Dense(units=16))
    model.add(Flatten())
    model.add(Dense(units=num_labels, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def create_lstm_model1(lr, dr, resolution, num_labels, num_hidden):
    curr_units = 2048

    model = Sequential()
    model.add(LSTM(units=curr_units, return_sequences=True, input_shape=(1, resolution)))

    for layer in range(num_hidden):
        curr_units = curr_units / 2
        model.add(LSTM(units=int(curr_units), return_sequences=True))
        model.add(Dropout(dr))

    curr_units = curr_units / 2
    model.add(LSTM(units=int(curr_units), return_sequences=False))
    model.add(Dense(units=36))
    model.add(Dense(units=16))
    model.add(Flatten())
    model.add(Dense(units=num_labels, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def create_gru_model(lr, dr, num_labels, num_hidden, in_shape):
    curr_units = 2048

    model = Sequential()
    model.add(GRU(units=curr_units, return_sequences=True, input_shape=in_shape))

    for layer in range(num_hidden):
        curr_units = curr_units / 2
        model.add(GRU(units=int(curr_units), return_sequences=True))
        model.add(Dropout(dr))

    curr_units = curr_units / 2
    model.add(GRU(units=int(curr_units), return_sequences=False))
    model.add(Dense(units=36))
    model.add(Dense(units=16))
    model.add(Flatten())
    model.add(Dense(units=num_labels, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return model


def create_cnn_model(lr, dr, num_labels, num_hidden, in_shape):
    curr_units = 256

    model = Sequential()
    model.add(Conv1D(filters=curr_units, kernel_size=7, input_shape=in_shape))  # 1
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=10))  # 2
    model.add(Dropout(dr))

    for layer in range(0, num_hidden):
        model.add(Conv1D(filters=int(curr_units), kernel_size=2))  # 3
        model.add(LeakyReLU())
        model.add(MaxPool1D(pool_size=2))  # 4
        model.add(Dropout(dr))

    model.add(Dense(16))
    model.add(Flatten())
    model.add(Dense(num_labels, activation='softmax'))  # 11

    opt = Adam(learning_rate=lr)
    model.compile(opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def create_dense_model(lr, dr, num_labels, in_shape):
    curr_units = 1200
    model = Sequential()
    model.add(Dense(curr_units, input_shape=in_shape))
    model.add(Dropout(dr))
    model.add(Dense(curr_units / 2))
    model.add(Dropout(dr))
    model.add(Dense(curr_units / 4))
    model.add(Dropout(dr))
    model.add(Dense(curr_units / 8))
    model.add(Dropout(dr))
    model.add(Dense(num_labels, activation='softmax'))

    opt = Adam(learning_rate=lr)
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    return model


def create_cnn_lstm_model(lr, dr, resolution, num_labels):
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=7, input_shape=(resolution, 4)))  # 1
    model.add(LeakyReLU())
    model.add(MaxPool1D(pool_size=10))  # 2
    model.add(Dropout(dr))
    model.add(Conv1D(filters=64, kernel_size=3))  # 3
    model.add(Dense(64))
    model.add(LSTM(units=10, return_sequences=True, input_shape=(32, 4)))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(num_labels, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model