import os 
import sys
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

from preprocess_human import load_filtered_data, split_into_epochs, identify_noisy_epochs, remove_duplicate_idx,calculate_psd, plot_psd, harmonic_filter, harmonic_filter_only


human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data/'
save_directory = '/home/melissa/RESULTS/HUMAN/harmonics_plots/'
patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                   'P29 N2', 'P30 N1']
                   
for patient in clean_patient_ids:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 5)
    print(number_epochs)
    channels_idx, epochs_idx, intercept_noise_df, slope_noise_df = identify_noisy_epochs(epochs, num_channels = 6, num_epochs = number_epochs)
    print(channels_idx)
    rm_dup = remove_duplicate_idx(intercept_noise_df, slope_noise_df)
    harmonic_indices = harmonic_filter(channels_idx, epochs_idx, epochs, rm_dup)
    all_noise = np.sort(np.append(harmonic_indices, rm_dup, axis=0))
    np.save('/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/' + str(patient) + '_noise.npy',  np.array(all_noise))
    power_concat = calculate_psd(epochs, all_noise, channels_idx, epochs_idx)
    print('power calculated')
    power_concat.to_csv('/home/melissa/RESULTS/HUMAN/power_df_harmonics/' + str(patient) + '_power_df.csv')
    plot_psd(power_concat, save_directory = save_directory, patient = patient)
    print('power plot saved')
       
        #harmonic_indices = harmonic_filter_only(number_epochs, epochs)
        
    
#Preprocess power filespower_folder = '/home/melissa/RESULTS/HUMAN/power_df_harmonics/'
power_save = '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Results/Human/Power/'

frequency_bands = [[1,4], [8, 13], [13, 30], [30, 35]]
frequency_names = ['delta', 'alpha', 'beta', 'gamma']

for patient in patient_list:
    print(patient)
    power_concat = pd.read_csv(f'{power_folder}{patient}_power_df.csv')
    epoch_ids = np.unique(power_concat['Epoch_IDX'])
    epoch_ls = []
    for idx in epoch_ids:
        epoch_df = power_concat.loc[power_concat['Epoch_IDX'] == idx]
        for channel in channel_ls:
            channel_df = epoch_df.loc[epoch_df['Channel'] == channel]
            epoch_freq_ls = []
            for freq_band, freq_name in zip(frequency_bands, frequency_names):
                freq_df = channel_df[(channel_df['Frequency'] >= freq_band[0]) & (channel_df['Frequency'] <= freq_band[1])]
                avg_freq = freq_df['Power'].mean()
                epoch_chan_df = pd.DataFrame(data = {f'Power_{freq_name}_{channel}': [avg_freq]})
                epoch_freq_ls.append(epoch_chan_df)
            freq_concat = pd.concat(epoch_freq_ls, axis = 1)
        epoch_id_df = pd.DataFrame(data = {'Patient': [patient], 'Epoch': [idx] })
        epoch_all_freqs = pd.concat([epoch_id_df, freq_concat], axis = 1)  
        epoch_ls.append(epoch_all_freqs)
    all_epochs = pd.concat(epoch_ls, axis = 0)
    all_epochs.to_csv(f'{power_save}{patient}_power.csv')