import serial
import keyboard
import time
import pandas as pd
import os

df_record = []
columns = ['timestamps', 'emg_signal']
directory = "Data/EMG"
msg_sent = True


def readserial(comport, baudrate):
    global df_record, columns, msg_sent
    ser = serial.Serial(comport, baudrate, timeout=0.1)  # 1/timeout is the frequency at which the port is read
    print("Waiting for EMG calibration...")
    while True:
        if keyboard.is_pressed("q"):
            qdata = pd.DataFrame(data=df_record, columns=columns)
            qdata.to_csv(f'{directory}/EMG{len(os.listdir(directory))}.csv', index=False)
            return
        data = ser.readline().decode().strip()
        if data:
            df_record.append((time.time(), data))
            if msg_sent:
                print("Started EMG Recording")
                msg_sent = False


if __name__ == '__main__':
    readserial('COM13', 115200)
