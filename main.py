import serial
import keyboard
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import timedelta


if __name__ == '__main__':

    for f in os.listdir("Data/Sync/Simple5"):
        if f.startswith("Simple5") and not f == "Simple5" and not f == "Simple1":
            df = pd.read_csv(f'Data/Sync/Simple5/{f}')
            plt.plot(df['emg_signal'])
    plt.show()
