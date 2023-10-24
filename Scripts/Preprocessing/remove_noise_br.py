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
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
seizure_br_path = '/home/melissa/PREPROCESSING/SYNGAP1/csv_seizures/'

train_2_ids =  ['S7096'] #['S7070', 'S7072', 'S7083', 'S7063', 'S7064', 'S7069']
train_1_ids = ['S7101']
seizure_two_syngap = ['S7063', 'S7064', 'S7069', 'S7072']
seizure_free_IDs = ['S7070', 'S7083', 'S7098', 'S7101', 'S7096']

br_number = 0

for animal in train_2_ids:
    animal = str(animal)
    print(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    #print(len(data_1[0]))
    #print(len(brain_state_1))
    #print(len(data_2[0]))
    #print(len(brain_state_2))
    #data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    print('all files loaded')
    if animal in seizure_free_IDs:
        explor_test_1 = FindNoiseThreshold(data = data_1, brain_state_file = brain_state_1,noise_limit = 3000, channelvariables = channel_variables, br_number = br_number)
        explor_test_2 = FindNoiseThreshold(data = data_2, brain_state_file = brain_state_2, noise_limit = 3000, channelvariables = channel_variables, br_number = br_number)
        clean_br_1, packet_loss_idx_1 = explor_test_1.find_packetloss_indices()
        clean_br_2, packet_loss_idx_2 = explor_test_2.find_packetloss_indices()
        slope_thresh_1, int_thresh_1 = explor_test_1.calc_noise_thresh(packet_loss_idx_1)
        slope_thresh_2, int_thresh_2 = explor_test_2.calc_noise_thresh(packet_loss_idx_2)
        noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'eeg')    
        noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2, channelvariables = channel_variables,ch_type = 'eeg')    
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
        power_calc_full_rec_1, noise_indices_1 = noise_filter_1.power_calc_noise(bandpass_filtered_data_1, slope_thresh = slope_thresh_1, int_thresh = int_thresh_1, clean_br = clean_br_1, br_number = br_number)
        power_calc_full_rec_2, noise_indices_2 = noise_filter_2.power_calc_noise(bandpass_filtered_data_2, slope_thresh = slope_thresh_2, int_thresh = int_thresh_2, clean_br = clean_br_2, br_number = br_number)
        
        
        os.chdir('/home/melissa/PREPROCESSING/SYNGAP1/clean_br/')
        clean_br_1.loc[noise_indices_1, 'brainstate'] = 5
        clean_br_2.loc[noise_indices_2, 'brainstate'] = 5
        clean_br_1.to_pickle(animal + '_BL1.pkl')
        clean_br_2.to_pickle(animal + '_BL2.pkl')
        
        power_calc_full_rec_1, noise_indices_1 = noise_filter_1.power_calc_noise(bandpass_filtered_data_1, slope_thresh = slope_thresh_1, int_thresh = int_thresh_1, clean_br = clean_br_1, br_number = br_number)
        power_calc_full_rec_2, noise_indices_2 = noise_filter_2.power_calc_noise(bandpass_filtered_data_2, slope_thresh = slope_thresh_2, int_thresh = int_thresh_2, clean_br = clean_br_2, br_number = br_number)    
        
        
    else:
        print('seizure pipeline beginning')
        os.chdir(seizure_br_path)
        br_1 = pd.read_csv( str(animal) + '_BL1_Seizures.csv')
        br_2 = pd.read_csv( str(animal) + '_BL2_Seizures.csv')
        br_no_seizures_1 = remove_seizure_epochs(brain_state_1, br_1)
        br_no_seizures_2 = remove_seizure_epochs(brain_state_2, br_2)
        print('removed seizure indices')
        explor_test_1 = FindNoiseThreshold(data = data_1, brain_state_file = br_no_seizures_1,noise_limit = 3000, channelvariables = channel_variables)
        explor_test_2 = FindNoiseThreshold(data = data_2, brain_state_file = br_no_seizures_2, noise_limit = 3000, channelvariables = channel_variables)
        clean_br_1, packet_loss_idx_1 = explor_test_1.find_packetloss_indices()
        clean_br_2, packet_loss_idx_2 = explor_test_2.find_packetloss_indices()
        slope_thresh_1, int_thresh_1 = explor_test_1.calc_noise_thresh(packet_loss_idx_1)
        slope_thresh_2, int_thresh_2 = explor_test_2.calc_noise_thresh(packet_loss_idx_2)
        noise_filter_1 = NoiseFilter(data_1, brain_state_file = br_no_seizures_1, channelvariables = channel_variables,ch_type = 'eeg')
        noise_filter_2 = NoiseFilter(data_2, brain_state_file = br_no_seizures_2, channelvariables = channel_variables,ch_type = 'eeg')
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
        power_calc_full_rec_1, noise_indices_1 = noise_filter_1.power_calc_noise(bandpass_filtered_data_1, slope_thresh = slope_thresh_1, int_thresh = int_thresh_1, clean_br = clean_br_1, br_number = br_number)
        power_calc_full_rec_2, noise_indices_2 = noise_filter_2.power_calc_noise(bandpass_filtered_data_2, slope_thresh = slope_thresh_2, int_thresh = int_thresh_2, clean_br = clean_br_2, br_number = br_number)
        
        clean_br_1.loc[noise_indices_1, 'brainstate'] = 5
        clean_br_2.loc[noise_indices_2, 'brainstate'] = 5
        
        print('all noise indices identified')
        harmonic_int_1 = HarmonicsFilter(filtered_data = bandpass_filtered_data_1, br_state_file = clean_br_1, br_state_num = br_number, noise_array = noise_indices_1)
        harmonic_indices_1 = harmonic_int_1.harmonics_algo()
        harmonic_int_2 = HarmonicsFilter(filtered_data = bandpass_filtered_data_2, br_state_file = clean_br_2, br_state_num = br_number, noise_array = noise_indices_2)
        harmonic_indices_2 = harmonic_int_2.harmonics_algo()
        print('harmonics analysis run')
    
        os.chdir('/home/melissa/PREPROCESSING/SYNGAP1/clean_br/')
        clean_br_1.loc[noise_indices_1, 'brainstate'] = 5
        clean_br_2.loc[noise_indices_2, 'brainstate'] = 5
        clean_br_1.loc[harmonic_indices_1, 'brainstate'] = 3
        clean_br_2.loc[harmonic_indices_2, 'brainstate'] = 3
        
        clean_br_1.to_pickle(animal + '_BL1.pkl')
        clean_br_2.to_pickle(animal + '_BL2.pkl')
        
        #power_calc_full_rec_1, noise_indices_1 = noise_filter_1.power_calc_noise(bandpass_filtered_data_1, slope_thresh = slope_thresh_1, int_thresh = int_thresh_1, clean_br = clean_br_1, br_number = br_number)
        #power_calc_full_rec_2, noise_indices_2 = noise_filter_2.power_calc_noise(bandpass_filtered_data_2, slope_thresh = slope_thresh_2, int_thresh = int_thresh_2, clean_br = clean_br_2, br_number = br_number)
        