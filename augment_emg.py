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
