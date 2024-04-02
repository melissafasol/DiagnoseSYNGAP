import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_hurst_exp

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'


patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1']  


# Iterate over each patient to process their data
for patient in patient_list:
    print(patient)
    # Generate the filename for each patient's data
    file_name = f'{patient}_(1).edf'
    # Load filtered data for the patient
    filtered_data = load_filtered_data(file_path=human_data_folder, file_name=file_name)
    # Split the data into epochs
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate=256, num_seconds=30)
    # Select indices of epochs that are considered clean
    clean_indices = select_clean_indices(noise_directory=noise_directory, patient_id=patient, total_num_epochs=number_epochs)
    
    print('Data loaded')
    # Generate a range of indices equal to the number of epochs
    all_indices = np.arange(number_epochs)
    
    # Initialize a dictionary to hold Hurst Exponent results for each channel
    hurst_results = {}
    # List of channels to compute Hurst for
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
    # Compute HFD for each channel using only clean indices
    for i, channel in enumerate(channels):
        hurst_results[channel] = [compute_hurst_exp(np.expand_dims(epochs[idx][i], axis=0))[0] for idx in clean_indices]
    
    print('All channels calculated')

    # Change directory to where results will be saved
    os.chdir(results_path)
    # Save HFD results for each channel
    for channel, data in hurst_results.items():
        np.save(f'{patient}_{channel}_hurst.npy', data)
    print(f'{patient} saved')
