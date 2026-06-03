# patternlockdecoding
IEEE publication: https://www.linkedin.com/safety/go/?url=https%3A%2F%2Fieeexplore%2Eieee%2Eorg%2Fdocument%2F10230156&urlhash=BaMC&mt=vdaLhiYL_K4lZgii9mQdb5ka_Ul1yziKYm6LnpkBpKNJnW0SCiwPMDPhUWcJVvMzU3ZmsBJWcPItlgkV3pKFbdhNf-g&isSdui=true

This repository has the software used for:
- Collecting data from the Muse2 EEG Headset and a set of EMG electrodes.
- Processing EEG and EMG data
- Creating deep learning models to learn on the data
- Running and evaluating the models

## Data Collection
***arduino_reader.py*** and ***muse_reader.py*** are run together using PyCharm Combined Configuration. Once muse_reader.py ends, it sends a signal to arduino_reader.py to stop recording.
arduino_reader.py - reads EMG data from Arduino on specified COM port at the selected baud rate and saves data into a single csv file once signal is received from muse_reader.py.

***muse_reader.py*** - reads EEG data from Muse2 Headset via Lab-Streaming Layer (LSL) and Bluetooth. This script also takes care of managing different pattern sessions and saves EEG data into .csv files based on which pattern is targeted. Sends signal to arduino_reader.py once session ends.

***data_synchronizer.py*** - matches chunks of EMG and EEG data based on timestamp and creates new files in "Sync" folder of merged EEG and EMG data - similar file structure as EEG save folder.

## Data

The finished dataset is made under the Data/Sync/ folder. This folder has subdirectories representing each pattern type and number. Types are Simple, MediumComplex, and Complex. The numbers are from the related work dataset on using Computer Vision to decode Pattern Locks: https://www.research.lancs.ac.uk/portal/en/datasets/pattern-lock-data-set-for-ndss17-paper-entitled-cracking-android-pattern-lock-in-five-attempts-version-2(fe7371b6-4c95-4cd4-8cf6-88e76ec6047d).html

Each .csv file contains both EEG and EMG data, along with timestamps used for syncing

***data_processing.py*** - processes the EEG and EMG data and returns dataset based on the type of experiment being run (see paper)

## Model Creation and Evaluation
***model_creation.py*** - creates various types of models via separate functions.

***model_evaluation.py*** - orchestrates the experiments. By default (unless script vars are configured) runs through ALL 3 experiments and all combinations. This means 5-fold cross validation for all models, and hyperparameter tuning combinations per experiment

***ml_metrics.py*** - stores any functions to visualize model performance i.e. confusion matrices.

## Misc

All other scripts are used for testing purposes or inspecting data.
