import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp
from mne_features.bivariate import compute_phase_lock_val, compute_max_cross_corr

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
#from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/CrossCor_Theta/'
 
SYNGAP_1_ID_ls =  ['S7101'] #[ 'S7088', 'S7092', 'S7094' , 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']
#SYNGAP_2_ID_ls = ['S7096']
SYNGAP_2_ID_ls =  ['S7096', 'S7070','S7072','S7083', 'S7063','S7064','S7069', 'S7086','S7091']


for animal in SYNGAP_1_ID_ls:
        print('one br running')
        print('loading ' + str(animal))
        animal = str(animal)
        load_files = LoadFiles(directory_path, animal)
        #data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
        print('data loaded')
        #only select eeg channels and filter with bandpass butterworth filter before selecting indices
        noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
        #noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        #bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
        print('data filtered')
        filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
        #filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
        cross_corr_ls_1 = []
        #cross_corr_ls_2 = []
        error_1 = []
        #error_2 = []
        
        for i in np.arange(0, 17280, 1):
            try:
                one_epoch_1 = compute_max_cross_corr(sfreq = 250.4, data = filter_1[:, i]) 
                cross_corr_ls_1.append(one_epoch_1)
            except:
                print(str(animal) + ' error for index ' + str(i))
                error_1.append(i)
            #try:
            #    one_epoch_2 = compute_max_cross_corr(sfreq = 250.4, data = filter_2[:, i])
            #    cross_corr_ls_2.append(one_epoch_2)
            #except:
            #    error_2.append(i)
            #    print(str(animal) + ' error for index ' + str(i))
        cross_corr_1 = np.array(cross_corr_ls_1)
        #cross_corr_2 = np.array(cross_corr_ls_2)
        error_array_1 = np.array(error_1)
        #error_array_2 = np.array(error_2)
        os.chdir(results_path)
        np.save(animal + '_cross_corr_1.npy', cross_corr_1)
        #np.save(animal + '_cross_corr_2.npy', cross_corr_2)
        np.save(animal + '_error_br_1.npy', error_array_1)
        #np.save(animal + '_error_br_2.npy', error_array_2)
        print('cross correlation values for ' + animal + ' saved')
    