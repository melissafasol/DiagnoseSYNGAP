import sys
import os 
import pandas as pd
import numpy as np 
import mne
from mne_connectivity import spectral_connectivity_time

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/mne_connectivity/'

frequency_bands = [[1,5], [5, 11], [11, 16], [16, 30], [30, 48]]
frequency_names = ['delta', 'theta', 'sigma', 'beta', 'gamma']
connectivity_measure = ['plv', 'pli', 'wpli'] #'coh'

SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091'] #S7101


analysis_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075',
               'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086', 'S7091', 'S7101']

for conn_mes in connectivity_measure:
    print(conn_mes)
    for animal in analysis_ls:
        print(animal)
        load_files = LoadFiles(directory_path, animal)
        if animal in syngap_2_ls:
            data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            print('data loaded')
            #only select eeg channels and filter with bandpass butterworth filter before selecting indices
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
            filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
            tr_filter_1  = filter_1.transpose(1, 0, 2)
            tr_filter_2 = filter_2.transpose(1, 0, 2)
            for freq_band, freq_name in zip(frequency_bands, frequency_names):
                connectivity_array_1 = spectral_connectivity_time(tr_filter_1, freqs = freq_band, n_cycles = 3,
                                                                  method= conn_mes,average = False, sfreq=250.4,
                                                                  faverage=True).get_data()
                connectivity_array_2 = spectral_connectivity_time(tr_filter_2, freqs = freq_band, n_cycles = 3,
                                                                  method= conn_mes,average = False, sfreq=250.4,
                                                                  faverage=True).get_data()
                connectivity_array = np.concatenate((connectivity_array_1, connectivity_array_2), axis=0)
                
                np.save(results_path + str(animal) + '_' + conn_mes + '_' + freq_name + '.npy', connectivity_array)
                
        elif animal in SYNGAP_1_ID_ls :
            data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
            tr_filter_1  = filter_1.transpose(1, 0, 2)
            for freq_band, freq_name in zip(frequency_bands, frequency_names):
                connectivity_array = spectral_connectivity_time(tr_filter_1, freqs = freq_band, n_cycles = 3,
                                                                method= conn_mes,average = False, sfreq=250.4,
                                                                faverage=True).get_data()
                np.save(results_path + str(animal) + '_' + conn_mes + '_' + freq_name + '.npy', connectivity_array)
        else:
            raise ValueError('Animal ID not found')
                