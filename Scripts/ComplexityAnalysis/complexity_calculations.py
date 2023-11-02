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
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

from complexity_class import ComplexClass

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Rat/Complexity/'


Somatosensory = [0, 6, 9, 13]
Motor = [1,2,3,10,11,12]
Visual = [4, 5, 7, 8]
region_ls = ['motor', 'somatosensory', 'visual']

SYNGAP_1_ID_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101']
syngap_2_ls =  ['S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086'] 

analysis_ls = [ 'S7088', 'S7092', 'S7094', 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075', 'S7101',
               'S7091', 'S7070', 'S7072', 'S7083', 'S7063','S7064', 'S7069', 'S7086']

analysis_ls_miss = ['S7063', 'S7064', 'S7069', 'S7088','S7086']

complex_calculation_types = ['hurst'] #hfd

for complx in complex_calculation_types:
    for animal in analysis_ls_miss:
        print('motor one starting')
        print('loading ' + str(animal))
        animal = str(animal)
        load_files = LoadFiles(directory_path, animal)
        if animal in syngap_2_ls:
            data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            print('data loaded')
            #only select eeg channels and filter with bandpass butterworth filter before selecting indices
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
            complexity_calculations = ComplexClass(complexity_calculation=complx, motor_indices=Motor, somatosensory_indices=Somatosensory,
                                               visual_indices=Visual)
            for region_type in region_ls:
                complex_array = complexity_calculations.two_recordings(bandpass_filtered_data_1= bandpass_filtered_data_1, 
                                                   bandpass_filtered_data_2= bandpass_filtered_data_2, region= region_type)
                np.save(results_path + complx + '/' + str(animal) + '_' + region_type + '.npy', complex_array)

            print('data saved')

        if animal in SYNGAP_1_ID_ls:
            data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
            noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
            bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
            complexity_calculations = ComplexClass(complexity_calculation=complx, motor_indices=Motor, somatosensory_indices=Somatosensory,
                                               visual_indices=Visual)
            for region_type in region_ls:
                complex_array = complexity_calculations.one_recording(bandpass_filtered_data_1= bandpass_filtered_data_1, region= region_type)
                np.save(results_path + complx + '/' + str(animal) + '_' + region_type + '.npy', complex_array)

            print('data saved')