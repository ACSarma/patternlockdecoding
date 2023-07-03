import datetime
import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.models import Sequential
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from keras.optimizers import Adam
import mne
from scipy import stats

import ml_metrics
import ml_models


def mean(data):
    return np.mean(data, axis=-1)


def std(data):
    return np.std(data, axis=-1)


def ptp(data):
    return np.ptp(data, axis=-1)


def var(data):
    return np.var(data, axis=-1)


def minim(data):
    return np.min(data, axis=-1)


def maxim(data):
    return np.max(data, axis=-1)


def argminim(data):
    return np.argmin(data, axis=-1)


def argmaxim(data):
    return np.argmax(data, axis=-1)


def mean_square(data):
    return np.mean(data ** 2, axis=-1)


def rms(data):  # root mean square
    return np.sqrt(np.mean(data ** 2, axis=-1))


def abs_diffs_signal(data):
    return np.sum(np.abs(np.diff(data, axis=-1)), axis=-1)


def skewness(data):
    return stats.skew(data, axis=-1)


def kurtosis(data):
    return stats.kurtosis(data, axis=-1)


def concatenate_features(data):
    return np.concatenate((mean(data),std(data),ptp(data),var(data),minim(data),maxim(data),argminim(data),argmaxim(data),
                          mean_square(data),rms(data),abs_diffs_signal(data),
                          skewness(data),kurtosis(data)),axis=-1)


print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))
# Use noise filtering for frequency early on
# Use statistical features from sklearn on all channels
# Reshape data for CNN implementation
directory = "Data/Sync"
type_dict = {
    "Simple": [
        1,
        5
    ],
    "Medium_Complex": [
        42,
        70
    ],
    "Complex": [
        108,
        119
    ]
}
resolution = 1296

X_data = []
Y_labels = []


if __name__ == '__main__':
    countlab = 0
    X_data = []
    Y_labels = []
    info = mne.create_info(ch_names=['AF7', 'AF8', 'TP9', 'TP10'], ch_types=['eeg', 'eeg', 'eeg', 'eeg'], sfreq=255)

    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            total_data.reverse()
            for file in total_data:
                try:
                    data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                    eeg = [data.AF7, data.AF8, data.TP9, data.TP10]
                    eeg = np.array(eeg)
                    eeg.resize(4, 1296)

                    raw = mne.io.RawArray(eeg, info)
                    raw.set_eeg_reference()
                    raw.filter(l_freq=1, h_freq=45, filter_length=1295)

                    X_data.append(raw.get_data())
                    Y_labels.append(1)

                except Exception as e:
                    print(e)
                    input()

            countlab = countlab + 1

    ntDir = "Data/EEG_NT"
    total_data = os.listdir(ntDir)
    for ntf in total_data:
        ntPath = f'{ntDir}/{ntf}'
        print(ntPath)
        total_path = os.listdir(ntPath)
        for file in total_path:
            try:
                print(f'{ntPath}/{file}')
                data = pd.read_csv(f'{ntPath}/{file}')
                eeg = [data.AF7, data.AF8, data.TP9, data.TP10]
                eeg = np.array(eeg)
                eeg.resize(4, 1296)

                for i in range(4):
                    noise = np.random.normal(loc=0, scale=15, size=1296)
                    aug_eeg = eeg
                    for j in range(4):
                        aug_eeg[j] = aug_eeg[j] + noise
                    raw = mne.io.RawArray(aug_eeg, info)
                    raw.set_eeg_reference()
                    raw.filter(l_freq=1, h_freq=45, filter_length=1295)
                    X_data.append(raw.get_data())
                    Y_labels.append(0)

                raw = mne.io.RawArray(eeg, info)
                raw.set_eeg_reference()
                raw.filter(l_freq=1, h_freq=45, filter_length=1295)

                X_data.append(raw.get_data())
                Y_labels.append(0)
            except Exception as e:
                print(e)
                input()

    Y_labels = np.array(Y_labels)
    X_data = np.array(X_data)
    X_data = np.moveaxis(X_data, 1, 2)
    # selected_funcs = {'mean', 'ptp_amp', 'std'}
    # X_data = concatenate_features(X_data)
    # X_data = np.reshape(X_data, (1020, 4212, 4))
    print(X_data.shape)
    print(Y_labels.shape)
    objects = StandardScaler()
    print(f'Unique Labels: {len(np.unique(Y_labels))}')
    print("Starting Model")
    gkf = StratifiedKFold(n_splits=5, shuffle=True)
    accuracy = []
    for train_index, val_index in gkf.split(X_data, Y_labels):
        train_features, train_labels = X_data[train_index], Y_labels[train_index]
        val_features, val_labels = X_data[val_index], Y_labels[val_index]
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features.reshape(-1, train_features.shape[-1])).reshape(
            train_features.shape)
        val_features = scaler.transform(val_features.reshape(-1, val_features.shape[-1])).reshape(val_features.shape)
        model = ml_models.create_cnn_model(resolution)
        model.fit(train_features, train_labels, epochs=50, batch_size=64, validation_data=(val_features, val_labels))
        accuracy.append(model.evaluate(val_features, val_labels)[1])

    # accuracies = []
    # summary = ""
    # drs = [0.4]  # dropout rates testing
    # lrs = [0.0001]  # learning rates testing
    # epochs = 200
    # best_hist = None
    #
    # # s = np.arange(0, len(X_data), 1)
    # # shuffle(s)
    # # X_data = X_data[s]
    # # Y_labels = Y_labels[s]
    # # GridSearch with Cross Validation
    # for r in range(len(drs)):
    #     for c in range(len(lrs)):
    #         kfolds = KFold(n_splits=5, shuffle=True, random_state=0)
    #         accuracies1 = []
    #         accuracies2 = []
    #         for train_mask, test_mask in kfolds.split(X_data, Y_labels):
    #             X_trainC = X_data[train_mask]
    #             y_trainC = Y_labels[train_mask]
    #
    #             X_testC = X_data[test_mask]
    #             y_testC = Y_labels[test_mask]
    #
    #             model2 = ml_models.create_cnn_model(lrs[c], drs[r], resolution)
    #
    #             callback = EarlyStopping(
    #                 monitor='sparse_categorical_accuracy', min_delta=0.0005,
    #                 patience=10)
    #
    #             hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
    #             y_predicted = model2.predict(X_testC)
    #             y_predicted_labels = [np.argmax(i) for i in y_predicted]
    #             acc = metrics.accuracy_score(y_testC, y_predicted_labels)
    #             print("Accuracy on Test: ", acc)
    #
    #             cm = confusion_matrix(y_testC, y_predicted_labels)
    #             ml_metrics.plot_confusion_matrix(cm, classes=range(10),
    #                                              title='')
    #             FP = cm.sum(axis=0) - np.diag(cm)
    #             FN = cm.sum(axis=1) - np.diag(cm)
    #             TP = np.diag(cm)
    #             TN = cm.sum() - (FP + FN + TP)
    #             FNR = FN / (TP + FN)
    #             TPR = TP / (TP + FN)
    #
    #             plt.show()
    #
    #             accuracies2.append(acc)
    #             if max(accuracies2) == acc:
    #                 best_hist = hist
    #
    #             summary += f'Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()}; FNR: {FNR}; TPR: {TPR} \n'
    #         print(accuracies2, "\nAverage Accuracy: ", np.average(accuracies2))
    #
    # e_list = []
    # for i in range(len(best_hist.history['loss'])):
    #     e_list.append(i)
    #
    # plt.plot(e_list, best_hist.history['loss'], label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.show()
    #
    # plt.plot(e_list, best_hist.history['sparse_categorical_accuracy'], label='Training Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.show()
    #
    # print(summary)
