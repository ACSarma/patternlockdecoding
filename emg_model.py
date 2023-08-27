import datetime
import os

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
    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            total_data.reverse()
            for file in total_data:
                try:
                    data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                    emg = data.emg_signal
                    emg = normalize([emg])[0]
                    emg = np.resize(emg, resolution)
                    emg = np.asarray(emg)
                    X_data.append(emg)
                    Y_labels.append(countlab)
                except Exception as e:
                    print(e)
            countlab = countlab + 1

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

    print(len(np.unique(Y_labels)))
    print("Starting Model")

    accuracies = []
    summary = ""
    summaryAvg = ""
    drs = [0, 0.2, 0.4]  # dropout rates testing
    lrs = [0.01, 0.001, 0.0001]  # learning rates testing
    hls = [0]
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

                    model2 = ml_models.create_lstm_model(lrs[c], drs[r], resolution, 6, hls[h])

                    callback = EarlyStopping(
                        monitor='sparse_categorical_accuracy', min_delta=0.001,
                        patience=10)

                    hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
                    y_predicted = model2.predict(X_testC)
                    y_predicted_labels = [np.argmax(i) for i in y_predicted]
                    acc = metrics.accuracy_score(y_testC, y_predicted_labels)
                    print("Accuracy on Test: ", acc)

                    cm = confusion_matrix(y_testC, y_predicted_labels)
                    # ml_metrics.plot_confusion_matrix(cm, classes=range(10),
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
