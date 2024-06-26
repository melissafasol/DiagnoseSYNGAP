import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data/'
results_path = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Human/CC/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'



def channel_pairs(channels):
        pairs = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                pairs.append([channels[i], channels[j]])
        return pairs

def calculate_max_cross_corr(patient_id , filtered_data, num_epochs, freq_band, channels):
        cross_corr_ls = []
        for i in np.arange(0, num_epochs, 1):
            print(i)
            one_epoch = compute_max_cross_corr(sfreq=256, data=filtered_data[i])
            df = pd.DataFrame([one_epoch], columns=[f'{ch[0]}_{ch[1]}_{freq_band}' for ch in channel_pairs(channels=channels)])
            df_other = pd.DataFrame(data = {'Epoch': [i], 'Patient_ID': [patient_id]})
            df_concat = pd.concat([df_other, df], axis = 1)
            cross_corr_ls.append(df_concat)
    

        cross_corr_concat = pd.concat(cross_corr_ls)
        return cross_corr_concat



frequency_bands = [[1,4], [8, 13], [13, 30], [30, 35]]
frequency_names = ['delta', 'alpha', 'beta', 'gamma']

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                   'P29 N2', 'P30 N1']  
channels = np.arange(0, 6, 1)

for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    all_freqs = []
    for freq_band, freq_name in zip(frequency_bands, frequency_names):
        filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name, l_freq = freq_band[0], h_freq= freq_band[1])
        number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 5)
        epoch_array = np.array(epochs)
        clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
        stacked_array = np.stack(epochs, axis=0)
        clean_array = stacked_array[clean_indices]
        cc_df = calculate_max_cross_corr(patient_id = patient, filtered_data = clean_array, num_epochs = len(clean_indices),
                                               freq_band = freq_name, channels = channels) 
        all_freqs.append(cc_df)

    cross_corr_concat = pd.concat(all_freqs)
    cross_corr_concat.to_csv(f'{results_path}{patient}_cross_corr.csv')
    print('cross correlation values for ' + patient + ' saved')
    