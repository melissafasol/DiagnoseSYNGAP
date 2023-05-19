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
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/Motor/HFD'


somatosensory = [0, 6, 9, 13]
Motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]

SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']
syngap_2_ls = ['S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091'] #S7101

for animal in SYNGAP_1_ID_ls:
    print('motor one starting')
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    #data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    print('data loaded')
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices
    noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
    #noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
    bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
    #bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
    print('data filtered')
    
    filter_1_1 = np.split(bandpass_filtered_data_1[1], 17280, axis = 0)
    filter_1_2 = np.split(bandpass_filtered_data_1[2], 17280, axis = 0)
    filter_1_3 = np.split(bandpass_filtered_data_1[3], 17280, axis = 0)
    filter_1_10 = np.split(bandpass_filtered_data_1[10], 17280, axis = 0)
    filter_1_11 = np.split(bandpass_filtered_data_1[11], 17280, axis = 0)
    filter_1_12 = np.split(bandpass_filtered_data_1[12], 17280, axis = 0)
    
    #filter_2_1 = np.split(bandpass_filtered_data_2[1], 17280, axis = 0)
    #filter_2_3 = np.split(bandpass_filtered_data_2[3], 17280, axis = 0)
    #filter_2_2 = np.split(bandpass_filtered_data_2[2], 17280, axis = 0)
    #filter_2_10 = np.split(bandpass_filtered_data_2[10], 17280, axis = 0)
    #filter_2_11 = np.split(bandpass_filtered_data_2[11], 17280, axis = 0)
    #filter_2_12 = np.split(bandpass_filtered_data_2[12], 17280, axis = 0)
    
    
    hfd_1_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_1]
    hfd_2_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_2]
    hfd_3_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_3]
    hfd_10_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_10]
    hfd_11_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_11]
    hfd_12_1 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_1_12]
    
    #hfd_1_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_1]
    #hfd_2_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_2]
    #hfd_3_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_3]
    #hfd_10_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_10]
    #hfd_11_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_11]
    #hfd_12_2 = [compute_higuchi_fd(np.expand_dims(epoch, axis = 0), kmax = 8) for epoch in filter_2_12]
   
    print('all channels calculated')
    mean_array_1 = np.mean(np.array([hfd_1_1, hfd_2_1, hfd_3_1, hfd_10_1, hfd_11_1, hfd_12_1]), axis = 0)
    #mean_array_2 = np.mean(np.array([hfd_1_2, hfd_2_2, hfd_3_2, hfd_10_2, hfd_11_2, hfd_12_2]), axis = 0)
    #mean_concat = np.concatenate((mean_array_1, mean_array_2), axis = 0)
    os.chdir(results_path)
    np.save(animal + '_hfd_concat.npy', mean_array_1)
    print(animal + ' saved')
    
    