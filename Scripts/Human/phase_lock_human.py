import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Phase_Lock'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = ['P27 N1'] 

            #['P23 N2', 'P23 N3', 'P21 N3']

                #['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1','P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1',
                # 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1', 'P23 N1', 'P24 N1',
                # 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1']


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    epoch_array = np.array(epochs)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
        
    phase_lock_ls = []
    error_1 = []
    
    for idx in clean_indices:
        try:
            reshape_data = np.moveaxis(np.moveaxis((epoch_array), 1, 0), 2,1)
            one_epoch_1 = compute_phase_lock_val(data = reshape_data[:, idx]) 
            phase_lock_ls.append(one_epoch_1)
        except:
            print(str(patient) + ' error for index ' + str(idx))
            error_1.append(idx)

    phase_lock = np.array(phase_lock_ls)
    os.chdir(results_path)
    np.save(patient + '_phase_lock.npy', phase_lock)
    error_array_1 = np.array(error_1)
    np.save(patient + '_error.npy', error_array_1)
    print('phase lock values for ' + patient + ' saved')
    