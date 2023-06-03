from muselsl import stream, list_muses
from muselsl import view
from muselsl import record
import csv
import os

type_dict = {
    # "Simple": [
    #     1,
    #     5
    # ],
    # "Medium_Complex": [
    #     42,
    #     70
    # ],
    "Complex": [
        108,
        # 119
    ]
}
samples_per_pattern = 10
directory = "Data/EEG"

if __name__ == '__main__':
    muses = list_muses()
    # view(version=2)

    for difficulty in type_dict.keys():
        for pattern in range(len(type_dict[difficulty])):
            sample_path = f'{directory}/{difficulty}{type_dict[difficulty][pattern]}/'
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)
            offset = len(os.listdir(sample_path))
            for i in range(samples_per_pattern):
                print(f'------------------ STARTING EEG RECORDING for {sample_path}{difficulty}{type_dict[difficulty][pattern]}_EEG{offset + i} --------------------')
                record(duration=6, filename=f'{sample_path}/{difficulty}{type_dict[difficulty][pattern]}_EEG{offset + i}.csv')

                # Note: Streaming is synchronous, so code here will not execute until after the stream has been closed
                print('Stream has ended')
