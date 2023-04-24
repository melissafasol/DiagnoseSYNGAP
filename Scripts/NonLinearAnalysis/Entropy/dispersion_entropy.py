import os 
import sys
import numpy as np 
import pandas as pd
import EntropyHub as EH


sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs


class DispersionEntropy:
    
    def __init__(self, dat_array):
        self.dat_array = dat_array
        
        
    def disp_en(self):
        disp_en_ls =  []

        for epoch in self.dat_array:
            Dispx_1, Ppi_1 = EH.DispEn(epoch, m = 3, tau = 2, c = 4, Typex = 'ncdf')
            disp_en_ls.append(Dispx_1)
        
        disp_en_array = np.array(disp_en_ls)
        
        return disp_en_array


directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
results_path = '/home/melissa/RESULTS/XGBoost/DispEn'


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
    disp_en_1 = DispersionEntropy(chan2_filter_1)
    dispen_array_1 = disp_en_1.disp_en() 
    print(str(animal) + 'dispersion entropy calculations br 1 complete')
    disp_en_2 = DispersionEntropy(chan2_filter_2)
    dispen_array_2 = disp_en_2.disp_en()
    print(str(animal) + 'dispersion entropy calculations br 2 complete')
    os.chdir(results_path)
    np.save(str(animal) + '_BR1.npy', dispen_array_1)
    np.save(str(animal) + '_BR2.npy', dispen_array_2)
    