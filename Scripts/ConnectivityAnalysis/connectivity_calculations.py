import os
import pandas as pd
import numpy as np 

rom mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

from connectivity_class import ConnectivityClass

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Connectivity'
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'


SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091'] #S7101

analysis_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']

connectivity_type = ['cross_corr', 'phase_lock_val'] 

for connectivity_cal in connectivity_type:
    for animal in analysis_ls:
        print('motor one starting')
        print('loading ' + str(animal))
        animal = str(animal)
        load_files = LoadFiles(directory_path, animal)
        if animal in syngap_2_ls:
            data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            print('data loaded')
            #only select eeg channels and filter with bandpass butterworth filter before selecting indices
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
            filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
            complexity_calculations_1 = ConnectivityClass(filter_1)
            complexity_calculations_2 = ConnectivityClass(filter_2)
            if connectivity_cal == 'cross_cor':
                max_cross_corr_1, error_1 = complexity_calculations_1.calculate_max_cross_corr()
                max_cross_corr_2, error_2 = complexity_calculations_2.calculate_max_cross_corr()
            elif connectivity_cal == 'phase_lock_val':
                phase_lock_val_1, error_1 = complexity_calculations_1.calculate_phase_lock_value()
                phase_lock_val_2, error_2 = complexity_calculations_2.calculate_phase_lock_value()

        if animal in SYNGAP_1_ID_ls :
            data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
            complexity_calculations_1 = ConnectivityClass(filter_1)
            if connectivity_cal == 'cross_cor':
                max_cross_corr_1, error_1 = complexity_calculations_1.calculate_max_cross_corr()
                np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '.npy')
                np.save(error_path + str(animal) + '_' + str(connectivity_cal) + '.npy')
            elif connectivity_cal == 'phase_lock_val':
                phase_lock_val_1, error_1 = complexity_calculations_1.calculate_phase_lock_value()
                np.save(results_path + str(connectivity_cal) + '/' + str(animal) + '_' + str(connectivity_cal) + '.npy')
                np.save(error_path + str(connectivity_cal) + '/' + str(animal) + '_' + str(connectivity_cal) + '.npy')