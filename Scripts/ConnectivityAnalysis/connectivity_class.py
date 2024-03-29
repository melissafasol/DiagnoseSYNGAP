import os
import numpy as np
import pandas as pd 
from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ls, SYNGAP_2_ls

#add feature to calculate by frequency band?

class ConnectivityClass:
    
    def __init__(self, filtered_data):
        self.filtered_data = filtered_data

    def calculate_max_cross_corr(self, num_epochs):
        cross_corr_ls = []
        error_ls = []
        for i in np.arange(0, num_epochs, 1):
            try:
                one_epoch_1 = compute_max_cross_corr(sfreq = 250.4, data = self.filtered_data[:, i]) 
                cross_corr_ls.append(one_epoch_1)
            except:
                print(' error for index ' + str(i))
                error_ls.append(i)
        
        cross_corr_array = np.array(cross_corr_ls)
        error_array = np.array(error_ls)
        
        return cross_corr_array, error_array
    
    def calculate_phase_lock_value(self, num_epochs):
        phase_lock_ls = []
        error_ls = []
        for i in np.arange(0, num_epochs, 1):
            try:
                one_epoch_1 = compute_phase_lock_val(sfreq = 250.4, data = self.filtered_data[:, i]) 
                phase_lock_ls.append(one_epoch_1)
            except:
                print(' error for index ' + str(i))
                error_ls.append(i)
        
        phase_lock_array = np.array(phase_lock_ls)
        error_array = np.array(error_ls)
        
        return phase_lock_array, error_array
    
    
def process_data(load_files, animal, connectivity_cal, results_path, is_single_file=False):
    # Load and process data
    if is_single_file:
        data, brain_state = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_list = [(data, brain_state)]
    else:
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        data_list = [(data_1, brain_state_1), (data_2, brain_state_2)]
    
    results = []
    for data, brain_state in data_list:
        noise_filter = NoiseFilter(data, brain_state_file=brain_state, channelvariables=channel_variables, ch_type='eeg')
        bandpass_filtered_data = noise_filter.filter_data_type()
        filtered_data = np.moveaxis(np.array(np.split(bandpass_filtered_data, 17280, axis=1)), 1, 0)
        complexity_calculations = ConnectivityClass(filtered_data)
        if connectivity_cal == 'cross_corr':
            result, error = complexity_calculations.calculate_max_cross_corr(num_epochs=17280)
        elif connectivity_cal == 'phase_lock_val':
            result, error = complexity_calculations.calculate_phase_lock_value(num_epochs=17280)
        results.append((result, error))
    
    # Save results
    if len(results) == 1:
        np.save(f'{results_path}{animal}_{connectivity_cal}.npy', results[0][0])
        np.save(f'{results_path}{animal}_{connectivity_cal}_error_1.npy', results[0][1])
    else:
        max_concat = np.concatenate([res[0] for res in results], axis=0)
        np.save(f'{results_path}{animal}_{connectivity_cal}.npy', max_concat)
        for i, (_, error) in enumerate(results, start=1):
            np.save(f'{results_path}{animal}_{connectivity_cal}_error_{i}.npy', error)
