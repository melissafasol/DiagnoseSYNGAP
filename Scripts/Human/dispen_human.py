import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 
import EntropyHub as EH

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/DispEn'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1']  

# Define a function to calculate DispEn for a channel
def calculate_dispen(patient_id, channel_data, clean_indices):
    dispen_values = [EH.DispEn(data, m=3, tau=2, c=4, Typex='ncdf')[0] for data in channel_data[clean_indices]]
    np.save(f"{patient_id}_{channel}_dispen.npy", dispen_values)

# Iterate through patient_list
for patient in patient_list:
    print(patient)
    file_name = f"{patient}_(1).edf"
    filtered_data = load_filtered_data(file_path=human_data_folder, file_name=file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate=256, num_seconds=30)
    clean_indices = select_clean_indices(noise_directory=noise_directory, patient_id=patient, total_num_epochs=number_epochs)
    print('Data loaded')
    
    # Calculate DispEn for each channel
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
    for channel in channels:
        calculate_dispen(patient, epochs[:, channels.index(channel)], clean_indices)
    
    print('All channels calculated')

    os.chdir(results_path)
    print(f"{patient} saved")
