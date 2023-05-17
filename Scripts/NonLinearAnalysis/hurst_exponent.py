import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
#from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Visual/Hurst'

Motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]

syngap_2_ls = ['S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091']

for animal in syngap_2_ls:
    print('visual starting')
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    print('data loaded')
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices
    noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
    noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
    bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
    bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
    print('data filtered')        
    filter_4_1 = np.array(np.split(bandpass_filtered_data_1[4], 17280, axis = 0))
    filter_5_1 = np.array(np.split(bandpass_filtered_data_1[5], 17280, axis = 0))
    filter_7_1 = np.array(np.split(bandpass_filtered_data_1[7], 17280, axis = 0))
    filter_8_1 = np.array(np.split(bandpass_filtered_data_1[8], 17280, axis = 0))
    
    filter_4_2 = np.array(np.split(bandpass_filtered_data_2[4], 17280, axis = 0))
    filter_5_2 = np.array(np.split(bandpass_filtered_data_2[5], 17280, axis = 0))
    filter_7_2 = np.array(np.split(bandpass_filtered_data_2[7], 17280, axis = 0))
    filter_8_2 = np.array(np.split(bandpass_filtered_data_2[8], 17280, axis = 0))
    
    hurst_4_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_4_1]
    hurst_5_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_5_1]
    hurst_7_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_7_1]
    hurst_8_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_8_1]
   
    hurst_4_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_4_2]
    hurst_5_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_5_2]
    hurst_7_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_7_2]
    hurst_8_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_8_2]
    
   
    print('all channels calculated')
    mean_array_1 = np.mean(np.array([hurst_4_1, hurst_5_1, hurst_7_1, hurst_8_1]), axis = 0)
    mean_array_2 = np.mean(np.array([hurst_4_2, hurst_5_2, hurst_7_2, hurst_8_2]), axis = 0)
    hurst_array = np.concatenate((mean_array_1, mean_array_2), axis = 0)
    os.chdir(results_path)
    np.save(animal + '_hurst_concat.npy', hurst_array)
    print(animal + ' saved')
    
    
    #filter_1_1 = np.array(np.split(bandpass_filtered_data_1[1], 17280, axis = 0))
    #filter_2_1 = np.array(np.split(bandpass_filtered_data_1[2], 17280, axis = 0))
    #filter_3_1 = np.array(np.split(bandpass_filtered_data_1[3], 17280, axis = 0))
    #filter_10_1 = np.array(np.split(bandpass_filtered_data_1[10], 17280, axis = 0))
    #filter_11_1 = np.array(np.split(bandpass_filtered_data_1[11], 17280, axis = 0))
    #filter_12_1 = np.array(np.split(bandpass_filtered_data_1[12], 17280, axis = 0))
    #
    #filter_1_2 = np.array(np.split(bandpass_filtered_data_2[1], 17280, axis = 0))
    #filter_2_2 = np.array(np.split(bandpass_filtered_data_2[2], 17280, axis = 0))
    #filter_3_2 = np.array(np.split(bandpass_filtered_data_2[3], 17280, axis = 0))
    #filter_10_2 = np.array(np.split(bandpass_filtered_data_2[10], 17280, axis = 0))
    #filter_11_2 = np.array(np.split(bandpass_filtered_data_2[11], 17280, axis = 0))
    #filter_12_2 = np.array(np.split(bandpass_filtered_data_2[12], 17280, axis = 0))
    #
    #hurst_1_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_1_1]
    #hurst_2_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_2_1]
    #hurst_3_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_3_1]
    #hurst_10_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_10_1]
    #hurst_11_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_11_1]
    #hurst_12_1 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_10_1]
    #
    #hurst_1_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_1_2]
    #hurst_2_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_2_2]
    #hurst_3_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_3_2]
    #hurst_10_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_10_2]
    #hurst_11_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_11_2]
    #hurst_12_2 = [compute_hurst_exp(np.expand_dims(epoch, axis = 0)) for epoch in filter_10_2]