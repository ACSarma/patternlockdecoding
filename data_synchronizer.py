import pandas as pd
import os
import os.path
from datetime import timedelta
import matplotlib.pyplot as plt

eeg_dir = "Data/EEG"
emg_dir = "Data/EMG"
lines_dir = "Data/Lines"
eeg_files = os.listdir(eeg_dir)
emg_files = os.listdir(emg_dir)
lines_files = os.listdir(lines_dir)
synced_dir = "Data/Sync"


def sync_lines(dfEMGsession):
    dfEMGsession = dfEMGsession.sort_values('timestamps')
    for subdir in os.listdir(lines_dir):
        subdirPath = f'{lines_dir}/{subdir}'
        dfPatterns = pd.read_csv(f'{subdirPath}/Session{len(os.listdir(subdirPath)) - 1}.csv')
        print(f'{subdirPath}/Session{len(os.listdir(subdirPath)) - 1}.csv')
        index = len(os.listdir(f'{synced_dir}/Lines/{subdir}'))
        for ind, row in dfPatterns.iterrows():
            if os.path.isfile(f'{synced_dir}/Lines/{subdir}/Lines{index}.csv'):
                index = index + 1
                continue
            else:
                sTime = (float(row['start']) * 0.001)
                eTime = float(row[' end']) * 0.001
                dfSub = dfEMGsession[(dfEMGsession['timestamps'] >= sTime) & (dfEMGsession['timestamps'] <= eTime)]
                if not dfSub.empty:
                    dfSub.to_csv(f'{synced_dir}/Lines/{subdir}/Lines{index}.csv')
                    index = index + 1


def sync_patterns(dfEMGsession):
    dfEMGsession = dfEMGsession.sort_values('timestamps')
    for subdir in os.listdir(lines_dir):
        subdirPath = f'{lines_dir}/{subdir}'
        dfPatterns = pd.read_csv(f'{subdirPath}/Session{len(os.listdir(subdirPath)) - 1}.csv')
        index = len(os.listdir(f'{synced_dir}/{subdir}'))
        for ind, row in dfPatterns.iterrows():
            if os.path.isfile(f'{synced_dir}/{subdir}/{subdir}_EMG{index}.csv'):
                index = index + 1
                continue
            else:
                sTime = (float(row['start']) * 0.001)
                eTime = float(row[' end']) * 0.001
                dfSub = dfEMGsession[(dfEMGsession['timestamps'] >= sTime) & (dfEMGsession['timestamps'] <= eTime)]
                if not dfSub.empty:
                    dfSub.to_csv(f'{synced_dir}/{subdir}/{subdir}_EMG{index}.csv')
                    index = index + 1


def sync_normal(dfEMG):
    for subdir in eeg_files:
        for eegF in os.listdir(f'{eeg_dir}/{subdir}'):
            print(f'Reading {eeg_dir}/{subdir}/{eegF}')
            if not os.path.isfile(f'{synced_dir}/{subdir}/{eegF}'):
                dfEEG = pd.read_csv(f'{eeg_dir}/{subdir}/{eegF}')
                dfEEG = dfEEG.sort_values('timestamps')
                dfEMG = dfEMG.sort_values('timestamps')
                df = pd.merge_asof(dfEEG, dfEMG, on="timestamps")
                df.to_csv(f'{synced_dir}/{subdir}/{eegF}')
            else:
                print("Skipping, already exists.")


if __name__ == '__main__':
    print(f'Reading: {emg_dir}/EMG{len(emg_files) - 1}.csv')
    df = pd.read_csv(f'{emg_dir}/EMG{len(emg_files) - 1}.csv')
    sync_lines(df)
    # sync_normal(df)
