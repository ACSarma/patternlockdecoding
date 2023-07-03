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
    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            X_data = []
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            total_data.reverse()
            for file in total_data:
                # try:
                data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                eeg = [data.AF7, data.AF8, data.TP9, data.TP10]
                eeg = np.array(eeg)
                eeg.resize(4, 1296)

                low = 0.5 / 128
                high = 45 / 128

                sos = butter(10, [low, high], btype='bandpass', output='sos')

                # data is a numpy array containing the samples of one EEG channel
                filtered = sosfilt(sos, eeg)
                info = mne.create_info(ch_names=['AF7', 'AF8', 'TP9', 'TP10'], ch_types=['eeg', 'eeg', 'eeg', 'eeg'], sfreq=255)
                raw = mne.io.RawArray(filtered, info)
                raw.set_eeg_reference()
                epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1)
                data = epochs.get_data()
                print(data.shape)
                plt.plot(eeg[0])
                plt.plot(filtered[0])
                plt.show()

                    # filtered is the filtered version of data
                    # emg = data.emg_signal
                    # emg = normalize([emg])[0]
                    # emg = np.resize(emg, resolution)
                    # emg = np.asarray(emg)
                    # X_data.append(emg)
                # except Exception as e:
                #     print(e)
                #     continue
            # objects = StandardScaler()
            # X_data = objects.fit_transform(X_data)
            # X_dataR = []
            # for emg in X_data:
            #     emg = np.resize(emg, (1, resolution))
            #     X_dataR.append(emg)
            # X_data = np.stack(X_dataR, axis=0)
            #
            # for i in range(1):
            #     rEMG = random.choice(X_data)[0]
            #     plt.plot(rEMG)
            # plt.title(f'Pattern Lock Type{countlab}')
            # ticks = [0, 200, 400, 600, 800, 1000, 1200]
            # ticklabels = [0, 1000, 2000, 3000, 4000, 5000, 6000]
            # plt.xticks(ticks, ticklabels)
            # plt.xlabel('Time (ms)')
            # plt.ylabel('Amplitude')
            # plt.show()
            # countlab = countlab + 1
