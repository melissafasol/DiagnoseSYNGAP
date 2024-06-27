import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 
from mne_connectivity import spectral_connectivity_time

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data/'
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Human/plv/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

frequency_bands = [[1,4], [8, 13], [13, 30], [30, 35]]
frequency_names = ['delta', 'alpha', 'beta', 'gamma']
connectivity_measure = ['plv']

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                   'P29 N2', 'P30 N1']  

def analyse_plv(array, freq_band, num_channels = 6, 
                    channel_labels = np.arange(0, 6, 1)):

        all_epochs = []
        for epoch in array:
            data_dict = {}

            # Ignore self-pairs in array
            for i in channel_labels:
                for j in channel_labels:
                    if i != j:
                        index = i * num_channels + j
                        pair_label = f"{i}_{j}_{freq_band}_plv"
                        data_dict[pair_label] = epoch[index]

            # Convert the dictionary to a DataFrame
            df = pd.DataFrame(data_dict, index=[0])  # Ensure each DataFrame has a consistent single row
            all_epochs.append(df)

        # Concatenate all DataFrames along the first axis
        df_concat = pd.concat(all_epochs, axis = 0, ignore_index=True)
        df_concat = df_concat.loc[:, (df_concat != 0).any(axis=0)]
        return df_concat


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 5)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    stacked_array = np.stack(epochs, axis=0)
    clean_array = stacked_array[clean_indices]
    all_freqs = []
    for freq_band, freq_name in zip(frequency_bands, frequency_names):
        connectivity_array = spectral_connectivity_time(clean_array, freqs = freq_band, method= 'plv',
                                          average = False, sfreq=256, n_cycles = 3, faverage=True).get_data()
        df_concat = analyse_plv(connectivity_array, freq_name, num_channels = 6, channel_labels = np.arange(0, 6, 1))
        all_freqs.append(df_concat)
    concat_all = pd.concat(all_freqs, axis = 1)
    print(concat_all)
    columns_to_drop = concat_all.columns[(concat_all == 0).all()]
    df = concat_all.drop(columns=columns_to_drop)
    df.to_csv(f'{results_path}{patient}_plv.csv')