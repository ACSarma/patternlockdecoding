import math
import random
import keyboard
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, StandardScaler
import numpy as np

directory = "Data/Sync"
type_dict = {
    # "Simple": [
    #     1,
    #     5
    # ],
    "Medium_Complex": [
        42,
        70
    ],
    "Complex": [
        108,
        119
    ]
}


def plot_list(in_list, title):
    legends = []
    ind = 0
    for item in in_list:
        plt.plot(item)
        legends.append("Sample" + str(ind))
        ind += 1
    plt.legend(legends)
    plt.title(title)
    plt.show()


count = 0
if __name__ == '__main__':
    # lines = os.listdir(f'{directory}/Lines')
    # lines.sort()
    # for li in lines:
    #     emg = pd.read_csv(f'{directory}/Lines/{li}/Lines{random.randint(0, 40)}.csv')
    #     emg = emg.emg_signal
    #     emg = normalize([emg])[0]
    #     plt.plot(emg)
    #     plt.legend(["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "L12", "L13", "L14", "L15", "L16"])
    # plt.show()

    for difficulty in type_dict.keys():
        for pattern in type_dict[difficulty]:
            total_data = os.listdir(f'{directory}/{difficulty}{pattern}')
            # total_data.reverse()
            emgs = []
            emgns = []
            for file in total_data:
                if count == 10:
                    print(f'{difficulty}{pattern}')
                    plot_list(emgs, "Raw EMG")
                    # X_data = np.stack(emgs, axis=0)
                    # objects = StandardScaler()
                    # X_data = objects.fit_transform(X_data)
                    # plot_list(abs(X_data))
                    for i in emgs:
                        print(i)
                        emgns.append(abs(normalize([i])[0]))
                    plot_list(emgns, "Normalized EMG")
                    count = 0
                    break
                count += 1
                data = pd.read_csv(f'{directory}/{difficulty}{pattern}/{file}')
                # eeg = data.AF8
                emg = data.emg_signal
                emg = np.asarray(emg, dtype=float)
                if np.isnan(emg).any():
                    print("found")
                    count -= 1
                else:
                    emgs.append(emg)

    for f in os.listdir("Data/Sync/"):
        if f.startswith("Complex108") and not f == "Complex108" and not f == "Complex119" and not f == "Medium_Complex70" and not f == "Medium_Complex42" and not f == "Simple5" and not f == "Simple1":
            df = pd.read_csv(f'Data/Sync/{f}')
            plt.plot(df['emg_signal'])
    plt.show()
