import os 
import sys
import numpy as np 
import pandas as pd
import scipy
from scipy import average, gradient, signal
import sys 
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/GRIN2B/scripts')
from GRIN2B_constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, br_animal_IDs, seizure_free_IDs, GRIN_het_IDs
from prepare_files import PrepareGRIN2B, LoadGRIN2B, GRIN2B_Seizures

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/taini_main/scripts/Preprocessing')
from preproc2_extractbrainstate import ExtractBrainStateIndices

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold


directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
seizure_br_path = '/home/melissa/PREPROCESSING/GRIN2B/seizures'
coherence_save_path = '/home/melissa/PREPROCESSING/GRIN2B/coherence/wake_wt_new/'
GRIN_wt_IDs = ['132', '362', '363', '375', '378', '382', '383', '140', '238',
                '240', '365', '366', '369', '371', '401', '402', '404', '430']
seizure_free_IDs = ['140', '228', '238', '362', '375']
br_number = 0

for animal in GRIN_wt_IDs:
    animal = str(animal)
    prepare = PrepareGRIN2B(directory_path, animal)
    recording, brain_state_1, brain_state_2 = prepare.load_two_analysis_files( seizure = 'False')
    start_time_1, start_time_2, end_time_1, end_time_2 = prepare.get_start_end_times(start_times_GRIN2B_dict = start_time_GRIN2B_baseline, end_times_GRIN2B_dict = end_time_GRIN2B_baseline)
    data_1 = recording[:, start_time_1: end_time_1 + 1]
    data_2 = recording[:, start_time_2: end_time_2 + 1]
    print('all files loaded')
    if animal in seizure_free_IDs:
        explor_test_1 = FindNoiseThreshold(data = data_1, br_number = br_number, brain_state_file = brain_state_1,noise_limit = 3000, channelvariables = channel_variables)
        explor_test_2 = FindNoiseThreshold(data = data_2, br_number = br_number, brain_state_file = brain_state_2, noise_limit = 3000, channelvariables = channel_variables)
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
        print('all noise indices identified')
        br_array_1 = coherence_arrays(br_state_num = br_number, br_state = clean_br_1, noise_indices = noise_indices_1, bandpass_filtered_data = bandpass_filtered_data_1)
        br_array_2 = coherence_arrays(br_state_num = br_number, br_state = clean_br_2, noise_indices = noise_indices_2, bandpass_filtered_data = bandpass_filtered_data_2)
    
        os.chdir(coherence_save_path)
        if br_array_1 is not None:
            np.save(str(animal) + '_wake_br_1.npy', br_array_1)
            print('dataframes for ' + str(animal) + ' saved')
        else:
            print('no epochs for 1 ' + str(animal))
        if br_array_2 is not None:
            np.save(str(animal) + '_wake_br_2.npy', br_array_2)
            print('dataframes for ' + str(animal) + ' saved')
        else:
            print('no epochs for 2 ' + str(animal))
    else:
        print('seizure pipeline beginning')
        os.chdir(seizure_br_path)
        br_1 = pd.read_csv('GRIN2B_' + str(animal) + '_BL1_Seizures.csv')
        br_2 = pd.read_csv('GRIN2B_' + str(animal) + '_BL2_Seizures.csv')
        br_no_seizures_1 = remove_seizure_epochs(brain_state_1, br_1)
        br_no_seizures_2 = remove_seizure_epochs(brain_state_2, br_2)
        print('removed seizure indices')
        explor_test_1 = FindNoiseThreshold(data = data_1, br_number = br_number, brain_state_file = br_no_seizures_1,noise_limit = 3000, channelvariables = channel_variables)
        explor_test_2 = FindNoiseThreshold(data = data_2, br_number = br_number, brain_state_file = br_no_seizures_2, noise_limit = 3000, channelvariables = channel_variables)
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
        print('all noise indices identified')
        
        harmonic_int_1 = HarmonicsFilter(filtered_data = bandpass_filtered_data_1, br_state_file = clean_br_1, br_state_num = br_number, noise_array = noise_indices_1)
        harmonic_indices_1 = harmonic_int_1.harmonics_algo()
        harmonic_int_2 = HarmonicsFilter(filtered_data = bandpass_filtered_data_2, br_state_file = clean_br_2, br_state_num = br_number, noise_array = noise_indices_2)
        harmonic_indices_2 = harmonic_int_2.harmonics_algo()
        
        br_array_1 = coherence_arrays(br_state_num = br_number, br_state = clean_br_1, noise_indices = harmonic_indices_1, bandpass_filtered_data = bandpass_filtered_data_1)
        br_array_2 = coherence_arrays(br_state_num = br_number, br_state = clean_br_2, noise_indices = harmonic_indices_2, bandpass_filtered_data = bandpass_filtered_data_2)
    
        os.chdir(coherence_save_path)
        if br_array_1 is not None:
            np.save(str(animal) + '_wake_br_1.npy', br_array_1)
            print('dataframes for ' + str(animal) + ' saved')
        else:
            print('no epochs for 1 ' + str(animal))
        if br_array_2 is not None:
            np.save(str(animal) + '_wake_br_2.npy', br_array_2)
            print('dataframes for ' + str(animal) + ' saved')
        else:
            print('no epochs for 2 ' + str(animal))