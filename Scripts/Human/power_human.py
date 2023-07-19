import os 
import sys
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing/')
from preprocess_human import load_filtered_data, split_into_epochs, identify_noisy_epochs, remove_duplicate_idx,calculate_psd, plot_psd, harmonic_filter, harmonic_filter_only

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data/'
save_directory = '/home/melissa/RESULTS/HUMAN/harmonics_plots/'
patient_list = [ 'P17 N2', 'P18 N2', 'P28 N2', 'P29 N1']
                
               #'P27 N1', 'P27 N2', 'P28 N1', 'P28 N2', 'P29 N1', 'P29 N2', 'P30 N1', 'P30 N2']  
                ##'P24 N1', 'P24 N2', 'P25 N1', 'P25 N2', 'P26 N1', 'P26 N2',

#'P1 N1', 'P1 N2', 'P2 N1', 'P2 N2', 'P10 N1', 'P10 N2', 'P11 N1', 'P11 N2', 'P12 N1','P12 N2', 'P13 N1',
#                'P13 N2', 'P15 N1', 'P15 N2', 'P16 N1', 'P16 N2', 'P17 N1', 'P17 N2', 'P18 N1', 'P18 N2', 'P19 N1', 'P19 N2', 'P20 N1', 'P20 N2', 'P21 N1', 'P21 N2', 'P22 N1',
#               'P22 N2', 'P23 N1', 'P23 N2'

#manual_list = ['P17 N2', 'P18 N2', 'P28 N2', 'P29 N1']


for patient in patient_list:
        print(patient)
        file_name = patient + '_(1).edf'
        filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
        number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
        print(number_epochs)
        try:
            channels_idx, epochs_idx, intercept_noise_df, slope_noise_df = identify_noisy_epochs(epochs, num_channels = 7, num_epochs = number_epochs)
            rm_dup = remove_duplicate_idx(intercept_noise_df, slope_noise_df)
            harmonic_indices = harmonic_filter(channels_idx, epochs_idx, epochs, rm_dup)
            all_noise = np.sort(np.append(harmonic_indices, rm_dup, axis=0))
            np.save('/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/' + str(patient) + '_noise.npy',  np.array(all_noise))
            power_concat = calculate_psd(epochs, all_noise, channels_idx, epochs_idx)
            print('power calculated')
            power_concat.to_csv('/home/melissa/RESULTS/HUMAN/power_df_harmonics/' + str(patient) + '_power_df.csv')
            plot_psd(power_concat, save_directory = save_directory, patient = patient)
            print('power plot saved')
        except:
            harmonic_indices = harmonic_filter_only(number_epochs, epochs)
        
    
    