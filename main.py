import serial
import keyboard
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

directory = "Data/Sync"
type_dict = {
    "Simple": [
        # 1,
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


count = 0
if __name__ == '__main__':
    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            total_data.reverse()
            for file in total_data:
                if count == 10:
                    plt.show()
                count += 1
                data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                eeg = data.AF8
                emg = data.emg_signal
                emg = normalize([emg])[0]
                plt.plot(emg)
    # for f in os.listdir("Data/Sync/"):
    #     if f.startswith("Complex108") and not f == "Complex108" and not f == "Complex119" and not f == "Medium_Complex70" and not f == "Medium_Complex42" and not f == "Simple5" and not f == "Simple1":
    #         df = pd.read_csv(f'Data/Sync/{f}')
    #         plt.plot(df['emg_signal'])
    # plt.show()
