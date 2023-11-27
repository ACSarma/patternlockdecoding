import datetime
import math
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
num_augmented = 5


def plot_list(in_list, pattern):
    legends = []
    ind = 0
    print(pattern)
    for item in in_list:
        plt.plot(item)
        legends.append("Sample" + str(ind))
        ind += 1
    plt.legend(legends)
    plt.show()


def getdata(resize=False, req_resolution=None):
    X_data = []
    Y_labels = []
    countlab = 0
    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            total_data.reverse()
            counting = 0
            for file in total_data:
                try:
                    data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                    emg = data.emg_signal
                    emg = np.resize(emg, resolution)
                    emg = np.asarray(emg)

                    snr = np.mean(emg) / np.std(emg)
                    power = np.sum(np.abs(emg)) / len(emg)
                    std = math.sqrt(power / snr)
                    for i in range(num_augmented):
                        noise = np.random.normal(0, std, len(emg))
                        new_sig = emg + noise
                        new_sig = np.asarray(new_sig)
                        new_sig = normalize([new_sig])[0]
                        X_data.append(new_sig)
                        Y_labels.append(countlab)
                    emg = normalize([emg])[0]
                    X_data.append(emg)
                    Y_labels.append(countlab)

                except Exception as e:
                    print(e)
                counting = counting + 1
            countlab = countlab + 1

    Y_labels = np.array(Y_labels)
    X_data = np.stack(X_data, axis=0)
    print(Y_labels.shape)
    print(len(np.unique(Y_labels)))
    objects = StandardScaler()
    X_data = objects.fit_transform(X_data)

    if resize:
        X_dataR = []
        for emg in X_data:
            emg = np.resize(emg, req_resolution)
            X_dataR.append(emg)
        X_data = np.stack(X_dataR, axis=0)

    print(X_data.shape)

    return X_data, Y_labels


if __name__ == '__main__':
    X_data, Y_labels = getdata(resize=False, req_resolution=(1, resolution))
    model = 'cnn'

    print("Starting Model")

    accuracies = []
    recalls = []
    summary = ""
    summaryAvg = ""
    drs = [0.0]  # dropout rates testing
    lrs = [0.001]  # learning rates testing
    hls = [1]
    epochs = 200
    best_hist = None
    best_cm = None

    # s = np.arange(0, len(X_data), 1)
    # shuffle(s)
    # X_data = X_data[s]
    # Y_labels = Y_labels[s]
    # GridSearch with Cross Validation
    for h in range(len(hls)):
        for r in range(len(drs)):
            for c in range(len(lrs)):
                kfolds = KFold(n_splits=5, shuffle=True, random_state=0)
                accuracies2 = []
                for train_mask, test_mask in kfolds.split(X_data, Y_labels):
                    X_trainC = X_data[train_mask]
                    y_trainC = Y_labels[train_mask]

                    X_testC = X_data[test_mask]
                    y_testC = Y_labels[test_mask]

                    if model == 'lstm':
                        model2 = ml_models.create_lstm_model(lrs[c], drs[r], 6, hls[h], (1, resolution))
                    if model == 'gru':
                        model2 = ml_models.create_gru_model(lrs[c], drs[r], 6, hls[h], (1, resolution))
                    if model == 'cnn':
                        model2 = ml_models.create_cnn_model(lrs[c], drs[r], 6, hls[h], (resolution, 1))
                    else:
                        model2 = ml_models.create_lstm_model(lrs[c], drs[r], 6, hls[h], (1, resolution))
                    dot_img_file = f'Models/{model}_emg.png'
                    tf.keras.utils.plot_model(model2, to_file=dot_img_file, show_shapes=True)
                    callback = EarlyStopping(
                        monitor='loss', min_delta=0.01,
                        patience=5)

                    hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
                    y_predicted = model2.predict(X_testC)
                    y_predicted_labels = [np.argmax(i) for i in y_predicted]
                    recalls.append(metrics.recall_score(y_testC, y_predicted_labels, average='macro'))
                    acc = metrics.accuracy_score(y_testC, y_predicted_labels)
                    print("Accuracy on Test: ", acc)

                    cm = confusion_matrix(y_testC, y_predicted_labels)

                    FP = cm.sum(axis=0) - np.diag(cm)
                    FN = cm.sum(axis=1) - np.diag(cm)
                    TP = np.diag(cm)
                    TN = cm.sum() - (FP + FN + TP)
                    FNR = FN / (TP + FN)
                    TPR = TP / (TP + FN)

                    accuracies2.append(acc)
                    if max(accuracies2) == acc:
                        best_hist = hist
                        best_cm = cm

                    summary += f'Hidden Layers: {hls[h]}, Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()}; FNR: {FNR}; TPR: {TPR} \n'
                print(accuracies2, "\nAverage Accuracy: ", np.average(accuracies2), "Hidden Layers: ", hls[h])
                summaryAvg += f'Hidden Layers: {hls[h]}, Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {np.average(accuracies2)}; Recall: {np.average(recalls)} - {datetime.datetime.now()} \n'

    e_list = []
    for i in range(len(best_hist.history['loss'])):
        e_list.append(i)

    print(summaryAvg)

    plt.plot(e_list, best_hist.history['loss'], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(e_list, best_hist.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    ml_metrics.plot_confusion_matrix(best_cm, classes=range(10),
                                     title='')
    plt.show()

