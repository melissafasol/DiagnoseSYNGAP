import os 
import sys
import pandas as pd 
import numpy as np
import scipy


import fooof
from fooof import FOOOF
from fooof.core.io import save_fm, load_json
from fooof.core.reports import save_report_fm

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs

from spec_slope_functions import SpectralSlope

directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
results_path = '/home/melissa/RESULTS/XGBoost/FOOOF/'

GRIN2B_ID_list = [ '138', '140', '402', '228', '238',
                 '363', '367', '378', '129', '137', '239', '383', '364'] 
                 #368, 132

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
    chan2_filter_1 = np.split(bandpass_filtered_data_1[2], 17280, axis = 0)
    chan2_filter_2 = np.split(bandpass_filtered_data_2[2], 17280, axis = 0)
    spec_slope_1 = SpectralSlope(chan2_filter_1)
    spec_slope_2 = SpectralSlope(chan2_filter_2)
    power_arr_1 = spec_slope_1.power_calc()
    power_arr_2 = spec_slope_2.power_calc()
    aperiodic_1 = spec_slope_1.fooof_analysis(power_arr_1)
    print(str(animal) + 'aperiodic br 1 complete')
    aperiodic_2 = spec_slope_2.fooof_analysis(power_arr_2)
    print(str(animal) + ' aperiodic br 2 complete')
    os.chdir(results_path)
    np.save(str(animal) + '_BR1.npy', power_arr_1)
    np.save(str(animal) + '_BR2.npy', power_arr_2)

    
    