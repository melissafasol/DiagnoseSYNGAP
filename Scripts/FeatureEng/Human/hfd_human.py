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
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Human/hfd/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

# List of patients to be processed
patient_list = [
    'P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
    'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1', 'P10 N1', 'P11 N1', 'P15 N1',
    'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
    'P22 N1', 'P23 N1', 'P23 N2', 'P23 N3', 'P24 N1', 'P27 N1', 'P28 N1',
    'P29 N2', 'P30 N1'
]

for patient in patient_list:
    print(f'Processing {patient}')
    file_name = f'{patient}_(1).edf'
    filtered_data = load_filtered_data(human_data_folder, file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, 256, 5)
    clean_indices = select_clean_indices(noise_directory, patient, number_epochs)

    # Compute both Hurst exponent and HFD for each channel
    results = {'hfd': {}}
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
    hfd_results = []
    for i, channel in enumerate(channels):
        hfd = [compute_higuchi_fd(np.expand_dims(epochs[idx][i], axis=0))[0] for idx in clean_indices]
        results['hfd'][channel] = hfd
        hfd_results.append(hfd)

    # Create DataFrame for HFD results
    df_hfd = pd.DataFrame(data=np.transpose(hfd_results), columns=[f'hfd_chan_{i}' for i in range(len(channels))])

    # Save the HFD results as a CSV file
    df_hfd.to_csv(f'{results_path}{patient}_hfd.csv', index=False)
    print(f'HFD data saved for {patient}')

