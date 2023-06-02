import serial
import keyboard
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta


if __name__ == '__main__':

    for f in os.listdir("Data/Sync/"):
        if f.startswith("Medium_Complex42") and not f == "Medium_Complex70" and not f == "Medium_Complex42" and not f == "Simple5" and not f == "Simple1":
            df = pd.read_csv(f'Data/Sync/{f}')
            plt.plot(df['emg_signal'])
    plt.show()
