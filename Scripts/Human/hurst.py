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
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Hurst'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = ['P1 N1', 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1',
             'P23 N1', 'P24 N1', 'P28 N1', 'P28 N2' 'P29 N2', 'P30 N1']


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    
    print('data loaded')
        
    hurst_chan_E1 = [compute_hurst_exp(np.expand_dims(epochs[idx][0], axis = 0)) for idx in clean_indices]
    hurst_chan_E2 = [compute_hurst_exp(np.expand_dims(epochs[idx][1], axis = 0)) for idx in clean_indices]
    hurst_chan_F3 = [compute_hurst_exp(np.expand_dims(epochs[idx][2], axis = 0)) for idx in clean_indices]
    hurst_chan_C3 = [compute_hurst_exp(np.expand_dims(epochs[idx][3], axis = 0)) for idx in clean_indices]
    hurst_chan_O1 = [compute_hurst_exp(np.expand_dims(epochs[idx][4], axis = 0)) for idx in clean_indices]
    hurst_chan_M2 = [compute_hurst_exp(np.expand_dims(epochs[idx][5], axis = 0)) for idx in clean_indices]
    
    print('all channels calculated')

    os.chdir(results_path)
    np.save(patient + '_E1_hurst.npy', hurst_chan_E1)
    np.save(patient + '_E2_hurst.npy', hurst_chan_E2)
    np.save(patient + '_F3_hurst.npy', hurst_chan_F3)
    np.save(patient + '_C3_hurst.npy', hurst_chan_C3)
    np.save(patient + '_01_hurst.npy', hurst_chan_O1)
    np.save(patient + '_M2_hurst.npy', hurst_chan_M2)
    print(patient + ' saved')