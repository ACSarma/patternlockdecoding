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
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils import shuffle
from keras.optimizers import Adam

print("Num GPUs Available", len(tf.config.experimental.list_physical_devices('GPU')))

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


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    print(cm)
    plt.colorbar()
    plt.clim(float(0.00), float(1.00))
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '%.2f' % (cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def create_model(lr, dr):
    regressor = Sequential()
    regressor.add(LSTM(units=128, return_sequences=True, input_shape=(1, resolution)))
    regressor.add(Dropout(dr))
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(dr))
    # regressor.add(Dropout(dr))
    # regressor.add(LSTM(units=128, return_sequences=True))
    # regressor.add(Dropout(dr))
    # regressor.add(LSTM(units=64, return_sequences=True))
    # regressor.add(Dropout(dr))
    # regressor.add(LSTM(units=32, return_sequences=True))
    regressor.add(Dropout(dr))
    regressor.add(Flatten())
    regressor.add(Dense(units=6, activation='sigmoid'))

    opt = Adam(learning_rate=lr)
    regressor.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['sparse_categorical_accuracy'],
    )

    return regressor


# def create_model(lr, dr):
#     mod = Sequential()
#     mod.add(Dense(2048))
#     mod.add(Dense(2048))
#     mod.add(Dense(1024))
#     mod.add(Dense(1024))
#     mod.add(Dropout(dr))
#     mod.add(Dense(512))
#     mod.add(Dense(256))
#     mod.add(Dropout(dr))
#     mod.add(Dense(256))
#     mod.add(Dense(128))
#     mod.add(Flatten())
#     mod.add(Dense(64))
#     mod.add(Dense(32))
#     mod.add(Dense(16))
#     mod.add(Dense(12, activation='sigmoid'))
#     # tf.keras.utils.plot_model(model, to_file='saved_model/mod.png', show_shapes=True)
#
#     opt = tf.keras.optimizers.SGD(learning_rate=lr)
#     mod.compile(
#         loss='sparse_categorical_crossentropy',
#         optimizer=opt,
#         metrics=['sparse_categorical_accuracy'],
#     )
#
#     return mod


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
                    emg = np.resize(emg, (resolution))
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
drs = [0.4]  # dropout rates testing
lrs = [0.0001]  # learning rates testing
epochs = 200
best_hist = None

# s = np.arange(0, len(X_data), 1)
# shuffle(s)
# X_data = X_data[s]
# Y_labels = Y_labels[s]
# GridSearch with Cross Validation
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

            model2 = create_model(lrs[c], drs[r])

            callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', min_delta=0.0005,
                patience=10)

            hist = model2.fit(X_trainC, y_trainC, epochs=200, callbacks=[callback])
            y_predicted = model2.predict(X_testC)
            y_predicted_labels = [np.argmax(i) for i in y_predicted]
            acc = metrics.accuracy_score(y_testC, y_predicted_labels)
            print("Accuracy on Test: ", acc)

            cm = confusion_matrix(y_testC, y_predicted_labels)
            plot_confusion_matrix(cm, classes=range(10),
                                  title='')
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            FNR = FN / (TP + FN)
            TPR = TP / (TP + FN)

            plt.show()

            accuracies2.append(acc)
            if max(accuracies2) == acc:
                best_hist = hist

            summary += f'Dropout: {drs[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()}; FNR: {FNR}; TPR: {TPR} \n'
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
