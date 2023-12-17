import math
import numpy as np


def return_augmented(data, num_augmented):
    for sample in data:
        snr = np.mean(sample) / np.std(sample)
        power = np.abs(sample)**2
        std = math.sqrt(power**2 / snr)
        for i in range(num_augmented):
            noise = np.random.normal(0, std, len(sample))
            new_sig = sample + noise
            data.append(new_sig)
    return data


'''
                    for i in range(num_augmented):
                        noise = np.random.normal(0, std, len(emg))
                        new_sig = emg + noise
                        # new_sig = np.asarray(new_sig)
                        # new_sig = normalize([new_sig])[0]
                        new_sig = np.resize(new_sig, len(data.emg_signal))
                        dataN = data.copy()
                        dataN['timestamps'] = dataN['timestamps'].astype(float)
                        dataN = dataN.round({'timestamps': 2})
                        dataN = dataN.loc[:, ~dataN.columns.str.contains('^Unnamed')]
                        dataN.emg_signal = new_sig
                        if "EMG" in file:
                            dataN.to_csv(f'{directory}/{difficulty}{pattern}/{difficulty}{pattern}_EMG{len(total_data) + acounter}.csv')
                            print(f'{difficulty}{pattern}_EMG{len(total_data) + acounter}.csv')
                        else:
                            dataN.to_csv(f'{directory}/{difficulty}{pattern}/{difficulty}{pattern}_EEG{len(total_data) + acounter}.csv')
                            print(f'{difficulty}{pattern}_EEG{len(total_data) + acounter}.csv')
                        acounter = acounter + 1
                        X_data.append(new_sig)
                        Y_labels.append(countlab)
'''
