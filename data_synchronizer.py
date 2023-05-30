import pandas as pd
import os
import matplotlib.pyplot as plt

eeg_dir = "Data/EEG"
emg_dir = "Data/EMG"
eeg_files = os.listdir(eeg_dir)
emg_files = os.listdir(emg_dir)
synced_dir = "Data/Sync"

if __name__ == '__main__':
    print(f'Reading: {emg_dir}/EMG{len(emg_files) - 1}.csv')
    dfEMG = pd.read_csv(f'{emg_dir}/EMG{len(emg_files) - 1}.csv')
    plt.plot(dfEMG['emg_signal'])
    plt.show()
    for subdir in eeg_files:
        offset = len(os.listdir(f'{eeg_dir}/{subdir}'))
        for eegF in os.listdir(f'{eeg_dir}/{subdir}'):
            print(f'Reading {eeg_dir}/{subdir}/{eegF}')

            dfEEG = pd.read_csv(f'{eeg_dir}/{subdir}/{eegF}')
            dfEEG = dfEEG.sort_values('timestamps')
            dfEMG = dfEMG.sort_values('timestamps')
            df = pd.merge_asof(dfEEG, dfEMG, on="timestamps")
            df.to_csv(f'{synced_dir}/{eegF}')
