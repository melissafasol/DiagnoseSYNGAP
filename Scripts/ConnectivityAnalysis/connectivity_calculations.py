import sys
import os
import pandas as pd
import numpy as np 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, SYNGAP_1_ls, SYNGAP_2_ls

from connectivity_class import ConnectivityClass, process_data

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Connectivity/'
error_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/ConnectivityErrors/'

analysis_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101',
                'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']

connectivity_type = ['cross_corr',  'phase_lock_val'] 

for connectivity_cal in connectivity_type:
    for animal in analysis_ls:
        print(f'Processing {animal} for {connectivity_cal}')
        load_files = LoadFiles(directory_path, animal)
        if animal in SYNGAP_2_ls:
            process_data(load_files, animal, connectivity_cal, is_single_file=False)
        elif animal in SYNGAP_1_ls:
            process_data(load_files, animal, connectivity_cal, is_single_file=True)

#for connectivity_cal in connectivity_type:
#    for animal in analysis_ls:
#        print(animal)
#        print('loading ' + str(animal))
#        animal = str(animal)
#        load_files = LoadFiles(directory_path, animal)
#        if animal in SYNGAP_2_ls:
#            data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
#            print('data loaded')
#            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
#            #only select eeg channels and filter with bandpass butterworth filter before selecting indices
#            noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
#            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
#            bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
#            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
 #           filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
 #           complexity_calculations_1 = ConnectivityClass(filter_1)
 #           complexity_calculations_2 = ConnectivityClass(filter_2)
 #           if connectivity_cal == 'cross_corr':
 #               print('cross corr beginning')
 #               max_cross_corr_1, error_1 = complexity_calculations_1.calculate_max_cross_corr(num_epochs= 17280)
 #               max_cross_corr_concat = np.concatenate((max_cross_corr_1, max_cross_corr_2), axis = 0)
 ##               max_cross_corr_2, error_2 = complexity_calculations_2.calculate_max_cross_corr(num_epochs= 17280)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '.npy', max_cross_corr_concat)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '_error_1.npy', error_1)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '_error_2.npy', error_2)
                
 #           elif connectivity_cal == 'phase_lock_val':
 #               print('phase lock beginning')
 #               phase_lock_val_1, error_1 = complexity_calculations_1.calculate_phase_lock_value(num_epochs= 17280)
 #               phase_lock_val_2, error_2 = complexity_calculations_2.calculate_phase_lock_value(num_epochs= 17280)
 #               max_phase_lock_concat = np.concatenate((phase_lock_val_1, phase_lock_val_2), axis = 0)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '.npy', max_phase_lock_concat)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '_error_2.npy', error_2)
 ##               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '_error_1.npy', error_1)
                

  #      if animal in SYNGAP_1_ls :
 #           print(animal + ' one file')
 #           data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
 #           noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
 #           bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
 #           filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
 #           complexity_calculations_1 = ConnectivityClass(filter_1)
 #           if connectivity_cal == 'cross_corr':
 #               print('cross corr beginning')
 #               max_cross_corr_1, error_1 = complexity_calculations_1.calculate_max_cross_corr(num_epochs= 17280 )
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '.npy', max_cross_corr_1)
 #               np.save(results_path + str(animal) + '_' + str(connectivity_cal) + '_error_1.npy', error_1)
 #           elif connectivity_cal == 'phase_lock_val':
 #               print('phase lock beginning')
 #               phase_lock_val_1, error_1 = complexity_calculations_1.calculate_phase_lock_value(num_epochs= 17280)
 #               np.save(results_path + str(connectivity_cal) + '/' + str(animal) + '_' + str(connectivity_cal) + '.npy', phase_lock_val_1)
 #               np.save(results_path + str(connectivity_cal) + '/' + str(animal) + '_' + str(connectivity_cal) + '_error_1.npy', error_1)
