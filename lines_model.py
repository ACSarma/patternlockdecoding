import datetime
import math
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, StandardScaler

import ml_metrics
import ml_models
import augment_emg

directory = "Data/Sync/Lines"

X_data = []
Y_labels = []
resolution = 100
num_augmented = 5

if __name__ == '__main__':
    lines = os.listdir(f'{directory}')
    for li in lines:
        label = int(re.search(r'\d+', li.title()).group())
        # if label % 8 == 0:
        line_data = f'{directory}/{li}'
        for sample in os.listdir(line_data):
            data = pd.read_csv(f'{line_data}/{sample}')
            emg = data.emg_signal
            emg = np.asarray(emg, dtype='float')
            emg = np.resize(emg, resolution)

            snr = np.mean(emg) / np.std(emg)
            power = np.sum(np.abs(emg)) / len(emg)
            std = math.sqrt(power / snr)
            for i in range(num_augmented):
                noise = np.random.normal(0, std, len(emg))
                new_sig = emg + noise
                new_sig = np.asarray(new_sig)
                X_data.append(new_sig)
                Y_labels.append(label)
                plt.plot(new_sig)
            # emg = normalize([emg])[0]
            X_data.append(emg)
            Y_labels.append(label)

            plt.plot(emg)
            plt.show()

    Y_labels = np.array(Y_labels)
    X_data = np.stack(X_data, axis=0)
    print(X_data.shape)
    print(Y_labels.shape)
    objects = StandardScaler()
    X_data = objects.fit_transform(X_data)
    X_dataR = []
    for emg in X_data:
        emg = np.resize(emg, (1, resolution))
        X_dataR.append(emg)
    X_data = np.stack(X_dataR, axis=0)

    unique_lab = len(np.unique(Y_labels))
    print("Starting Model")

    accuracies = []
    summary = ""
    summaryAvg = ""
    drs = [0, 0.4]  # dropout rates testing
    lrs = [0.0005]  # learning rates testing
    hls = [1, 2, 3]
    epochs = 200
    best_hist = None

    # s = np.arange(0, len(X_data), 1)
    # shuffle(s)
    # X_data = X_data[s]
    # Y_labels = Y_labels[s]
    # GridSearch with Cross Validation
    for h in range(len(hls)):
        for r in range(len(drs)):
            for c in range(len(lrs)):
                kfolds = KFold(n_splits=5, shuffle=True, random_state=0)
                accuracies1 = []
                accuracies2 = []
                for train_mask, test_mask in kfolds.split(X_data, Y_labels):
                    X_trainC = X_data[train_mask]
                    y_trainC = Y_labels[train_mask]

                    X_testC = X_data[test_mask]
                    y_testC = Y_labels[test_mask]

                    model2 = ml_models.create_lstm_model(lrs[c], drs[r], resolution, unique_lab, hls[h])

                    callback = EarlyStopping(
                        monitor='sparse_categorical_accuracy', min_delta=0.001,
                        patience=10)

                    hist = model2.fit(X_trainC, y_trainC, epochs=epochs, callbacks=[callback])
                    y_predicted = model2.predict(X_testC)
                    y_predicted_labels = [np.argmax(i) for i in y_predicted]
                    acc = metrics.accuracy_score(y_testC, y_predicted_labels)
                    print("Accuracy on Test: ", acc)

                    cm = confusion_matrix(y_testC, y_predicted_labels)
                    # ml_metrics.plot_confusion_matrix(cm, classes=range(unique_lab),
                    #                       title='')
                    FP = cm.sum(axis=0) - np.diag(cm)
                    FN = cm.sum(axis=1) - np.diag(cm)
                    TP = np.diag(cm)
                    TN = cm.sum() - (FP + FN + TP)
                    FNR = FN / (TP + FN)
                    TPR = TP / (TP + FN)

                    # plt.show()

                    accuracies2.append(acc)
                    if max(accuracies2) == acc:
                        best_hist = hist

                    summary += f'Hidden Layers: {hls[h]}, Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()}; FNR: {FNR}; TPR: {TPR} \n'
                print(accuracies2, "\nAverage Accuracy: ", np.average(accuracies2), "Hidden Layers: ", hls[h])
                summaryAvg += f'Hidden Layers: {hls[h]}, Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {np.average(accuracies2)} - {datetime.datetime.now()} \n'

    e_list = []
    for i in range(len(best_hist.history['loss'])):
        e_list.append(i)

    plt.plot(e_list, best_hist.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(e_list, best_hist.history['sparse_categorical_accuracy'], label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    print(summaryAvg)
