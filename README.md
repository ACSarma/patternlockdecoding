# patternlockdecoding

This repository has the software used for collecting data from the Muse2 EEG Headset and a set of EMG electrodes.

main.py - does nothing, just used as a sandbox for testing functionalities, viewing data for now.

arduino_reader.py - reads EMG data from Arduino on specified COM port at the selected baud rate and saves data into a single csv file once "q" is pressed on keyboard

muse_reader.py - reads EEG data from Muse2 Headset via Lab-Streaming Layer (LSL) and Bluetooth. This script also takes care of managing different pattern sessions and saves EEG data into .csv files based on which pattern is targeted

data_synchronizer.py - matches chunks of EMG and EEG data based on timestamp and creates new files in "Sync" folder of merged EEG and EMG data - similar file structure as EEG save folder.

