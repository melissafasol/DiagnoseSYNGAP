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
#from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

from spec_slope_functions import SpectralSlope

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/FOOOF_all_channels'

syngap_2_ls = ['S7063'] #, 'S7072', 'S7083', 'S7064', 'S7069', 'S7086', 'S7091'] #['S7063']
SYNGAP_1_ID_ls = ['S7098'] #, 'S7092', 'S7094'] #, 'S7098, [S7088', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']


channel_indices = [0, 1, 2, 3,4, 5,6,7,8,9,10, 11,12,13]  #0, 1, 2, 3, 4, 5, 6, 
#chanel_indices = [1, 7, 8, 9, 10,11,12,13]

for animal in SYNGAP_1_ID_ls:
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
        for chan_idx in channel_indices:
                print(chan_idx)
                filter_1 = np.split(bandpass_filtered_data_1[chan_idx], 17280, axis = 0)
                #filter_2 = np.split(bandpass_filtered_data_2[chan_idx], 17280, axis = 0)
                spec_slope_1 = SpectralSlope(filter_1)
                #spec_slope_2 = SpectralSlope(filter_2)
                power_arr_1 = spec_slope_1.power_calc()
                #power_arr_2 = spec_slope_2.power_calc()
                offset_1, exponent_1, error_1 = spec_slope_1.fooof_analysis(power_arr_1)
                print(str(animal) + ' channel ' + str(chan_idx) + ' aperiodic br 1 complete')
                #offset_2, exponent_2, error_2 = spec_slope_2.fooof_analysis(power_arr_2)
                #print(str(animal) + ' channel ' + str(chan_idx) + ' aperiodic br 2 complete')
                os.chdir(results_path)
                np.save(str(animal) + '_' + str(chan_idx) + '_' + 'offset_BR1.npy', offset_1)
                np.save(str(animal) + '_' + str(chan_idx) + '_' + 'exponent_BR1.npy', exponent_1)
                #np.save(str(animal) + '_' + str(chan_idx) + '_'+ 'offset_BR2.npy', offset_2)
                #np.save(str(animal) + '_' + str(chan_idx) + '_' + 'exponent_BR2.npy', exponent_2)
                np.save(str(animal) + '_' + str(chan_idx) + '_' + 'error_1.npy', error_1)
                #np.save(str(animal) + '_' + str(chan_idx) + '_' + 'error_2.npy', error_2)
                
        
        

    
    