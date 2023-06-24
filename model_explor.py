import datetime

import serial
import keyboard
from time import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

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


def create_model(lr, dr):
    regressor = Sequential()
    regressor.add(LSTM(units=512, return_sequences=True, input_shape=(1, 1296)))
    regressor.add(LSTM(units=256, return_sequences=True))
    regressor.add(LSTM(units=128, return_sequences=True))
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(dr))
    regressor.add(Flatten())
    regressor.add(Dense(units=6, activation='softmax'))

    opt = Adam(learning_rate=lr)
    regressor.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return regressor


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
                    eeg = data.AF8
                    emg = data.emg_signal
                    emg = normalize([emg])[0]
                    emg = np.asarray(emg)
                    emg = np.resize(emg, (1, 1296))
                    X_data.append(emg)
                    Y_labels.append(countlab)
                except Exception as e:
                    print(e)
            countlab = countlab + 1

    Y_labels = np.array(Y_labels)
    X_data = np.stack(X_data, axis=0)
    print(X_data.shape)
    print(Y_labels.shape)
print(len(np.unique(Y_labels)))
print("Starting Model")

accuracies = []
summary = ""
num_train = 2
drs = [0.05, 0.2, 0.4]  # dropout rates testing
lrs = [0.001, 0.0001]  # learning rates testing
epochs = 200
best_hist = None

s = np.arange(0, len(X_data), 1)
shuffle(s)
X_data = X_data[s]
Y_labels = Y_labels[s]
# GridSearch with Cross Validation
for r in range(len(drs)):
    for c in range(len(lrs)):
        kfolds = KFold(n_splits=2, shuffle=True, random_state=0)
        accuracies1 = []
        accuracies2 = []
        for train_mask, test_mask in kfolds.split(X_data, Y_labels):
            X_trainC = X_data[train_mask]
            y_trainC = Y_labels[train_mask]

            X_testC = X_data[test_mask]
            y_testC = Y_labels[test_mask]

            model2 = create_model(lrs[c], drs[r])

            callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', min_delta=0.0001,
                patience=10)

            hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
            y_predicted = model2.predict(X_testC)
            y_predicted_labels = [np.argmax(i) for i in y_predicted]
            acc = metrics.accuracy_score(y_testC, y_predicted_labels)
            print("Accuracy on Test: ", acc)

            accuracies2.append(acc)
            if max(accuracies2) == acc:
                best_hist = hist

            summary += f'Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()} \n'
        print(accuracies2, "\nAverage Accuracy: ", np.average(accuracies2))

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

print(summary)
