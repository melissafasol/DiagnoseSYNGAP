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
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs

directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
results_path = '/home/melissa/RESULTS/XGBoost/Visual/HFD/'

GRIN2B_ID_list = ['131', '229', '236', '237', '241', '362', '366', '368', 
                    '369', '373', '132', '138', '140', '402', '228', '238',
                    '363', '367', '378', '129', '137', '239', '383', '364',
                    '365', '371', '382', '404', '130', '139', '401', '240',
                    '227', '375', '424', '433', '430'] 

somatosensory = [0, 6, 9, 13]
Motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]

for animal in GRIN2B_ID_list:
    print('visual starting')
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = start_time_GRIN2B_baseline, end_times_dict = end_time_GRIN2B_baseline)
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
    
    
    hfd_4_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_4_1]
    hfd_5_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_5_1]
    hfd_7_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_7_1]
    hfd_8_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_8_1]
    
    hfd_4_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_4_2]
    hfd_5_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_5_2]
    hfd_7_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_7_2]
    hfd_8_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_8_2]
   
    print('all channels calculated')
    mean_array_1 = np.mean(np.array([hfd_4_1, hfd_5_1, hfd_7_1, hfd_8_1]), axis = 0)
    mean_array_2 = np.mean(np.array([hfd_4_2, hfd_5_2, hfd_7_2, hfd_8_2]), axis = 0)
    mean_concat = np.concatenate((mean_array_1, mean_array_2), axis = 0)
    os.chdir(results_path)
    np.save(animal + '_hfd_concat.npy', mean_concat)
    print(animal + ' saved')
    
    