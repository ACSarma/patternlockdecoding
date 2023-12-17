'''
Author: Arun Sarma
This file processes EEG and EMG data. It contains functions to return labelled data for each of the 3 experiments.
This is used by model_evaluation.py
'''

import os
import re

import mne
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler

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
directory_lines = "Data/Sync/Lines"


def get_emg_data(req_resolution=None):
    X_data = []
    Y_labels = []
    countlab = 0
    resolution = 1296
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

    if req_resolution is not None:
        X_dataR = []
        for emg in X_data:
            emg = np.resize(emg, req_resolution)
            X_dataR.append(emg)
        X_data = np.stack(X_dataR, axis=0)

    print(X_data.shape)

    return X_data, Y_labels


def get_eeg_data(reshape=False):
    resolution = 1275
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
                    eeg.resize(4, 1275)

                    raw = mne.io.RawArray(eeg, info)
                    raw.set_eeg_reference()
                    raw.filter(l_freq=1, h_freq=45, filter_length=resolution)

                    X_data.append(eeg)
                    Y_labels.append(1)
                except Exception as e:
                    print(e)
                    print(f'{directory}/{difficulty}{pattern}/{file}')

    ntDir = "Data/EEG_NT"
    total_data = os.listdir(ntDir)
    for ntf in total_data:
        ntPath = f'{ntDir}/{ntf}'
        total_path = os.listdir(ntPath)
        for file in total_path:
            try:
                print(f'{ntPath}/{file}')
                data = pd.read_csv(f'{ntPath}/{file}')
                eeg = [data.AF7, data.AF8, data.TP9, data.TP10]
                eeg = np.array(eeg)
                print(eeg.shape)
                if eeg.shape[1] < 1275:
                    eeg.resize(4, 1275)
                    print("resized")
                    print(eeg.shape)
                raw = mne.io.RawArray(eeg, info)
                raw.set_eeg_reference()
                raw.filter(l_freq=1, h_freq=45)
                epochs = mne.make_fixed_length_epochs(raw, duration=5)
                eeg_pro = epochs.get_data()

                for sig in eeg_pro:
                    X_data.append(sig)
                    Y_labels.append(0)
            except Exception as e:
                print(e)
                continue

    Y_labels = np.array(Y_labels)
    X_data = np.array(X_data)

    if reshape:
        X_data = np.moveaxis(X_data, 1, 2)

    print(X_data.shape)
    print(Y_labels.shape)

    print(f'Unique Labels: {len(np.unique(Y_labels))}')

    return X_data, Y_labels


def get_emg_lines_data(req_resolution=None):
    resolution = 100
    X_data = []
    Y_labels = []
    lines = os.listdir(f'{directory_lines}')
    for li in lines:
        label = int(re.search(r'\d+', li.title()).group())
        line_data = f'{directory_lines}/{li}'
        for sample in os.listdir(line_data):
            data = pd.read_csv(f'{line_data}/{sample}')
            emg = data.emg_signal
            emg = np.asarray(emg, dtype='float')
            emg = np.resize(emg, resolution)
            X_data.append(emg)
            Y_labels.append(label)

    Y_labels = np.array(Y_labels)
    X_data = np.stack(X_data, axis=0)
    print(X_data.shape)
    print(Y_labels.shape)
    unique_labs = len(np.unique(Y_labels))
    print(unique_labs)
    objects = StandardScaler()
    X_data = objects.fit_transform(X_data)

    if req_resolution is not None:
        X_dataR = []
        for emg in X_data:
            emg = np.resize(emg, req_resolution)
            X_dataR.append(emg)
        X_data = np.stack(X_dataR, axis=0)

    return X_data, Y_labels, unique_labs
