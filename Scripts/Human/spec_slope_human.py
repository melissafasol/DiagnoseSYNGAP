import os 
import scipy 
import numpy as np
import pandas as pd 

import fooof
from fooof import FOOOF
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/FOOOF_all_channels/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = ['P1 N1', 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1',
             'P23 N1', 'P24 N1', 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1']


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    
    channels_idx = list(np.arange(0, 6))
    for chan in channels_idx:
        power_ls = []
        for clean_idx in clean_indices:
            power_calculations = scipy.signal.welch(epochs[clean_idx][chan], window = 'hann', fs = 256, nperseg = 7680)
            power = power_calculations[1][9:1051]
            power_ls.append(power)

        power_array = np.array(power_ls)
        frequency_range = [0.3, 35]
        frequency_values = np.arange(0.3, (35 + 0.03333333333333333), 0.03333333333333333)
        
        offset_ls = []
        exponent_ls = []
        error_idx = []
        
        for i, epoch in enumerate(power_array):
            try: 
                fm = FOOOF()
                fm.report(frequency_values, epoch, frequency_range)
                aperiodic_values = fm.aperiodic_params_
                offset_ls.append(aperiodic_values[0])
                exponent_ls.append(aperiodic_values[1])
            except:
                print('error at index ' + str(i))
                error_idx.append(i)
                
            
        offset_array = np.array(offset_ls)
        exponent_array = np.array(exponent_ls)
        error_array = np.array(error_idx)
        
        np.save(results_path + str(patient) + '_chan_' + str(chan) + '_offset.npy', offset_array)
        np.save(results_path + str(patient) + '_chan_' + str(chan) + '_exponent.npy', exponent_array)
        np.save(results_path + str(patient) + '_chan_' + str(chan) + '_error.npy', error_array)
        