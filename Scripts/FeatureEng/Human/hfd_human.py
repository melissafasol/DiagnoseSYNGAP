import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/HFD_All_Epochs/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

# List of patients to be processed
patient_list = [
    'P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
    'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1', 'P10 N1', 'P11 N1', 'P15 N1',
    'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
    'P22 N1', 'P23 N1', 'P23 N2', 'P23 N3', 'P24 N1', 'P27 N1', 'P28 N1',
    'P28 N2', 'P29 N2', 'P30 N1'
]

# Process data for each patient
for patient in patient_list:
    print(f'Processing {patient}')
    file_name = f'{patient}_(1).edf'
    filtered_data = load_filtered_data(human_data_folder, file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, 256, 30)
    clean_indices = select_clean_indices(noise_directory, patient, number_epochs)

    # Compute both Hurst exponent and HFD for each channel
    results = {'hurst': {}, 'hfd': {}}
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
    for i, channel in enumerate(channels):
        hurst = [compute_hurst_exp(np.expand_dims(epochs[idx][i], axis=0)) for idx in np.arange(number_epochs)]
        hfd = [compute_higuchi_fd(np.expand_dims(epochs[idx][i], axis=0))[0] for idx in clean_indices]
        results['hurst'][channel] = hurst
        results['hfd'][channel] = hfd

    # Change to the results directory and save the results
    os.chdir(results_path)
    for analysis_type, channels_data in results.items():
        for channel, data in channels_data.items():
            np.save(f'{patient}_{channel}_{analysis_type}.npy', data)
    print(f'{patient} data saved')