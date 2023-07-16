import itertools
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pandas as pd
from mne.io import BaseRaw
from scipy.signal import butter, sosfilt
from sklearn.preprocessing import normalize, StandardScaler
import mne

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
resolution = 1200

if __name__ == '__main__':
    countlab = 0
    X_data = []
    Y_labels = []
    info = mne.create_info(ch_names=['AF7', 'AF8', 'TP9', 'TP10'], ch_types=['eeg', 'eeg', 'eeg', 'eeg'], sfreq=255)

    # ntDir = "Data/EEG_NT"
    # total_data = os.listdir(ntDir)
    # for ntf in total_data:
    #     ntPath = f'{ntDir}/{ntf}'
    #     print(ntPath)
    #     total_path = os.listdir(ntPath)
    #     for file in total_path:
    #         try:
    #             print(f'{ntPath}/{file}')
    #             data = pd.read_csv(f'{ntPath}/{file}')
    #             eeg = [data.AF7, data.AF8, data.TP9, data.TP10]
    #             eeg = np.array(eeg)
    #             eeg.resize(4, 1296)
    #
    #             for i in range(4):
    #                 noise = np.random.normal(loc=0, scale=15, size=1296)
    #                 aug_eeg = eeg
    #                 for j in range(4):
    #                     aug_eeg[j] = aug_eeg[j] + noise
    #                 raw = mne.io.RawArray(aug_eeg, info)
    #                 raw.set_eeg_reference()
    #                 raw.filter(l_freq=1, h_freq=45, filter_length=1295)
    #                 X_data.append(raw.get_data())
    #                 Y_labels.append(0)
    #
    #             raw = mne.io.RawArray(eeg, info)
    #             raw.set_eeg_reference()
    #             raw.filter(l_freq=1, h_freq=45, filter_length=1295)
    #
    #             X_data.append(raw.get_data())
    #             Y_labels.append(0)
    #         except Exception as e:
    #             print(e)
    #             input()

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
                    Y_labels.append(countlab)

                except Exception as e:
                    print(e)
                    input()

            countlab = countlab + 1

    X_data = np.array(X_data)
    print(X_data.shape)
    while True:
        elem = random.choice(X_data)
        fig, axs = plt.subplots(4)
        ticks = [0, 200, 400, 600, 800, 1000, 1200]
        ticklabels = [0, 1000, 2000, 3000, 4000, 5000, 6000]
        axs[0].plot(elem[0])
        axs[0].xticks(ticks, ticklabels)
        axs[1].plot(elem[1])
        axs[1].xticks(ticks, ticklabels)
        axs[2].plot(elem[2])
        axs[2].xticks(ticks, ticklabels)
        axs[3].plot(elem[3])
        axs[3].xticks(ticks, ticklabels)
        plt.show()
