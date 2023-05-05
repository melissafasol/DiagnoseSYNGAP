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
results_path = '/home/melissa/RESULTS/XGBoost/HFD/'

GRIN2B_ID_list = ['131', '229', '236', '237', '241', '362', '366', '368', 
                    '369', '373', '132', '138', '140', '402', '228', '238',
                    '363', '367', '378', '129', '137', '239', '383', '364',
                    '365', '371', '382', '404', '130', '139', '401', '240',
                    '227', '375', '424', '433', '430'] 

for animal in GRIN2B_ID_list:
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
    chan2_filter_1 = np.array(np.split(bandpass_filtered_data_1[2], 17280, axis = 0))
    chan2_filter_2 = np.array(np.split(bandpass_filtered_data_2[2], 17280, axis = 0))
    hfd_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in chan2_filter_1]
    hfd_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in chan2_filter_2]
    hfd_array_1 = np.array(hfd_1)
    hfd_array_2 = np.array(hfd_2)
    hfd_array = np.concatenate((hfd_array_1, hfd_array_2), axis = 0)
    os.chdir(results_path)
    np.save(animal + '_hfd_concat.npy', hfd_array)
    print(animal + ' saved')
    
    