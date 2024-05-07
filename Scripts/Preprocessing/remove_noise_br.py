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
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, seizure_free_IDs, seizure_two_files, SYNGAP_1_ID_ls, SYNGAP_2_ID_ls

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
seizure_br_path = '/home/melissa/PREPROCESSING/SYNGAP1/csv_seizures/'
clean_br_path = '/home/melissa/PREPROCESSING/SYNGAP1/clean_br/'

br_number = 0

def preprocess_data_2_animals(animal_ids, directory_path, seizure_free_ids, seizure_br_path, SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, br_number):
    for animal_id in animal_ids:
        print(animal_id)
        load_files = LoadFiles(directory_path, animal_id)
        data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        print('all files loaded')

        # Determine if the animal is seizure-free
        seizure_free = animal_id in seizure_free_ids

        # Preprocess brain state files based on seizure presence
        if not seizure_free:
            print('seizure pipeline beginning')
            br_1 = pd.read_csv(seizure_br_path + f'{animal_id}_BL1_Seizures.csv')
            br_2 = pd.read_csv(seizure_br_path + f'{animal_id}_BL2_Seizures.csv')
            brain_state_1 = remove_seizure_epochs(brain_state_1, br_1)
            brain_state_2 = remove_seizure_epochs(brain_state_2, br_2)
            print('removed seizure indices')

        # Common processing for all animals
        noise_filter_1 = NoiseFilter(data_1, brain_state_file=brain_state_1, channelvariables=channel_variables, ch_type='eeg')
        noise_filter_2 = NoiseFilter(data_2, brain_state_file=brain_state_2, channelvariables=channel_variables, ch_type='eeg')
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
        bandpass_filtered_data_2 = noise_filter_2.filter_data_type()

        # Find noise and calculate thresholds
        processed_data_1 = process_data(noise_filter_1, bandpass_filtered_data_1, br_number)
        processed_data_2 = process_data(noise_filter_2, bandpass_filtered_data_2, br_number)

        # Save cleaned brain states
        save_clean_brain_states(clean_br_path, animal_id, processed_data_1['clean_br'], processed_data_2['clean_br'])
        
        
def preprocess_data_1_animal(animal_ids, directory_path, seizure_free_ids, seizure_br_path, SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, br_number):
    for animal_id in animal_ids:
        print(animal_id)
        load_files = LoadFiles(directory_path, animal_id)
        data_1, brain_state_1 = load_files.load_one_analysis_file(start_times_dict=SYNGAP_baseline_start, end_times_dict=SYNGAP_baseline_end)
        print('all files loaded')

        # Determine if the animal is seizure-free
        seizure_free = animal_id in seizure_free_ids

        # Preprocess brain state files based on seizure presence
        if not seizure_free:
            print('seizure pipeline beginning')
            br_1 = pd.read_csv(seizure_br_path + f'{animal_id}_BL1_Seizures.csv')
            brain_state_1 = remove_seizure_epochs(brain_state_1, br_1)
            print('removed seizure indices')

        # Common processing for all animals
        noise_filter_1 = NoiseFilter(data_1, brain_state_file=brain_state_1, channelvariables=channel_variables, ch_type='eeg')
        bandpass_filtered_data_1 = noise_filter_1.filter_data_type()

        # Find noise and calculate thresholds
        processed_data_1 = process_data(noise_filter_1, bandpass_filtered_data_1, br_number)

        # Save cleaned brain states
        save_clean_brain_states(clean_br_path, animal_id, processed_data_1['clean_br'])

def process_data(noise_filter, bandpass_filtered_data, br_number, animal_id, seizure_free_ids):
    clean_br, packet_loss_idx = noise_filter.find_packetloss_indices()
    slope_thresh, int_thresh = noise_filter.calc_noise_thresh(packet_loss_idx)
    power_calc_full_rec, noise_indices = noise_filter.power_calc_noise(bandpass_filtered_data, slope_thresh=slope_thresh, int_thresh=int_thresh, clean_br=clean_br, br_number=br_number)

    # Determine if the animal is seizure-free
    seizure_free = animal_id in seizure_free_ids

    # Mark noise indices
    clean_br.loc[noise_indices, 'brainstate'] = 5
    if not seizure_free:
        harmonic_indices = HarmonicsFilter(filtered_data=bandpass_filtered_data, br_state_file=clean_br, br_state_num=br_number, noise_array=noise_indices).harmonics_algo()
        print('harmonics analysis run')
        clean_br.loc[harmonic_indices, 'brainstate'] = 3

    return {'clean_br': clean_br, 'noise_indices': noise_indices}

def save_clean_brain_states(clean_br_path, animal_id, clean_br_1, clean_br_2):
    clean_br_1.to_pickle(clean_br_path + f'{animal_id}_BL1.pkl')
    clean_br_2.to_pickle(clean_br_path + f'{animal_id}_BL2.pkl')
    print('Saved cleaned brain states')
    
#input list of animal ids to preprocess 
preprocess_data_2_animals(SYNGAP_2_ID_ls, directory_path, seizure_free_IDs, seizure_br_path, SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables, br_number)