'''Coherence preprocessing for Jorge's code'''

import os 
import sys
import numpy as np 
import pandas as pd
import scipy
from scipy import average, gradient, signal
import sys 
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, channel_variables
from coherence import coherence_arrays

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/NonLinearAnalysis/ImagCoherence')
from coherence import reshape_coherence_array

directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
seizure_br_path = '/home/melissa/PREPROCESSING/GRIN2B/seizures'
coherence_save_path = '/home/melissa/PREPROCESSING/GRIN2B/coherence/seizures_wt/'
GRIN_wt_IDs = ['132', '362', '363', '375', '378', '382', '383', '140', '238',
               '240', '365', '366', '369', '371', '401', '402', '404', '430']
seizure_free_IDs = ['140', '228', '238', '362', '375']
br_number = 4

for animal in GRIN_wt_IDs:
    animal = str(animal)
    if animal in seizure_free_IDs:
        pass
    else:
        print('loading ' + animal)
        load_files = LoadFiles(directory_path, animal)
        recording_1, recording_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = start_time_GRIN2B_baseline, 
                                                                                                end_times_dict = end_time_GRIN2B_baseline)
        explor_test_1 = FindNoiseThreshold(data = recording_1, br_number = br_number, brain_state_file = brain_state_1,noise_limit = 3000, channelvariables = channel_variables)
        explor_test_2 = FindNoiseThreshold(data = recording_2, br_number = br_number, brain_state_file = brain_state_2, noise_limit = 3000, channelvariables = channel_variables)
        clean_br_1, packet_loss_idx_1 = explor_test_1.find_packetloss_indices()
        clean_br_2, packet_loss_idx_2 = explor_test_2.find_packetloss_indices()
        noise_filter_1 = NoiseFilter(recording_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
        noise_filter_2 = NoiseFilter(recording_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
        os.chdir(seizure_br_path)
        seizure_br_1 = pd.read_csv('GRIN2B_' + str(animal) + '_BL1_Seizures.csv')
        seizure_br_2 = pd.read_csv('GRIN2B_' + str(animal) + '_BL2_Seizures.csv')
        seizure_br_included_1 = remove_seizure_epochs(clean_br_1, seizure_br_1)
        seizure_br_included_2 = remove_seizure_epochs(clean_br_2, seizure_br_2)
        seizure_epochs_1 = load_files.extract_br_state(bandpass_filtered_data_1, seizure_br_included_1, br_number)
        seizure_epochs_2 = load_files.extract_br_state(bandpass_filtered_data_2, seizure_br_included_2, br_number)
        reshape_array_1 = reshape_coherence_array(seizure_epochs_1)
        reshape_array_2 = reshape_coherence_array(seizure_epochs_2)
        np.save(coherence_save_path + str(animal) + '_seiz_br_1.npy', reshape_array_1)
        np.save(coherence_save_path + str(animal) + '_seiz_br_2.npy', reshape_array_2)
        print('files saved for ' + animal)