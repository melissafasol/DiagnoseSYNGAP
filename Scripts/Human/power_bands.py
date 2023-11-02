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
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/Delta_Power/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = [ 'P27 N1']
    
    #'P23 N2', 'P23 N3', 'P21 N3'

#['P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1','P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1']

            #['P1 N1', 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1',
            # 'P23 N1', 'P24 N1', 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1']


#indices for frequency bands
# delta [15:61] [1:4]
# theta [120: 211] [4:8]
# alpha [140: 361] [8:12]
# beta [450: 901] [13:30]


for patient in patient_list:
    print(patient)
    print('delta')
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    
    channels_idx = list(np.arange(0, 6))
    for chan in channels_idx:
        power_ls = []
        for clean_idx in clean_indices:
            power_calculations = scipy.signal.welch(epochs[clean_idx][chan], window = 'hann', fs = 256, nperseg = 7680)
            avg_power = np.mean(power_calculations[1][15:61])
            power_ls.append(avg_power)

        power_array = np.array(power_ls)
        np.save(results_path + str(patient) + '_chan_' + str(chan) + '.npy', power_array)