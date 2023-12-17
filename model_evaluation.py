'''
Author: Arun Sarma
Course: CSS595
Purpose: This script orchestrates model evaluation for all 3 experiments based on configurable variables in the script.
Returns model training logs and metrics of models.
'''

import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

import ml_metrics
import model_creation
import data_processing

print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))

'''
Specify which experiment or models to run by commenting/uncommenting list entries below:
'''
experiments = [
    'PLIdent_Patterns',
    'PLIdent_Lines',
    'PLDetect'
]
models = ['cnn',
          'lstm',
          'gru'
          ]

resolution = None


def get_data(exp, model):
    global resolution
    if exp == "PLIdent_Patterns":
        resolution = 1296
        if model == 'cnn':
            return data_processing.get_emg_data()
        return data_processing.get_emg_data(req_resolution=(1, resolution))
    if exp == "PLIdent_Lines":
        resolution = 100
        if model == 'cnn':
            return data_processing.get_emg_lines_data()
        return data_processing.get_emg_lines_data(req_resolution=(1, resolution))
    if exp == "PLDetect":
        resolution = 1275
        if model == 'cnn':
            return data_processing.get_eeg_data(reshape=True)
        return data_processing.get_eeg_data(reshape=False)


if __name__ == '__main__':
    for exp in experiments:
        for model in models:
            print(f'############ Starting test of Experiment: {exp} with Model: {model}. ###############')

            X_data, Y_labels = get_data(exp, model)

            accuracies = []
            recalls = []
            summary = ""
            summaryAvg = ""
            drs = [0.0, 0.2, 0.4]  # dropout rates testing
            lrs = [0.01, 0.001, 0.0001]  # learning rates testing
            hls = [1]
            epochs = 200
            best_hist = None
            best_cm = None

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
                                model2 = model_creation.create_lstm_model(lrs[c], drs[r], 6, hls[h], (1, resolution))
                            if model == 'gru':
                                model2 = model_creation.create_gru_model(lrs[c], drs[r], 6, hls[h], (1, resolution))
                            if model == 'cnn':
                                model2 = model_creation.create_cnn_model(lrs[c], drs[r], 6, hls[h], (resolution, 1))
                            else:
                                model2 = model_creation.create_lstm_model(lrs[c], drs[r], 6, hls[h], (1, resolution))

                            callback = EarlyStopping(
                                monitor='loss', min_delta=0.01,
                                patience=5)

                            hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
                            y_predicted = model2.predict(X_testC)
                            y_predicted_labels = [np.argmax(i) for i in y_predicted]
                            recalls.append(metrics.recall_score(y_testC, y_predicted_labels, average='macro'))
                            acc = metrics.accuracy_score(y_testC, y_predicted_labels)
                            print("Accuracy on Test: ", acc)

                            ml_metrics.plot_loss(hist)

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
