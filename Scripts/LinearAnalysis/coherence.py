'''Script to save coherence arrays in the correct format for running on Jorge's GUI'''

import os 
import numpy as np
import pandas as pd

from exploratory import FindNoiseThreshold
from filter import NoiseFilter
from parameters import channelvariables



def coherence_arrays(br_state_num, br_state, noise_indices, bandpass_filtered_data):
    br_state.loc[noise_indices, 'brainstate'] = 5
    br_state_selected = br_state.loc[br_state['brainstate'] == br_state_num]
    if len(br_state_selected) > 0:
        start_times = br_state_selected['start_epoch'].to_numpy()
        br_epochs = []
        for start in start_times:
            start_samp = int(start*250.4)
            end_samp = start_samp + 1252
            one_epoch = bandpass_filtered_data[:, start_samp:end_samp]
            br_epochs.append(one_epoch)
        br_array = np.array(br_epochs)
    
        #Reshape array to flatten epochs for coherence analysis 
        arr = np.moveaxis(br_array, 1, -1)
        br_arr = arr.reshape(-1, *arr.shape[-1:]).T
        return br_arr
    else:
        pass


#Necessary File Paths 
directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
seizure_br_path = '/home/melissa/PREPROCESSING/GRIN2B/seizures'
coherence_save_path = '/home/melissa/PREPROCESSING/GRIN2B/coherence/NREM_het/'
GRIN_het_IDs = ['131', '130', '129', '228', '227', '229', '373', '138', '137',
                '139','236', '237', '239', '241', '364', '367', '368', '424',
                '433']


for animal in GRIN_het_IDs:
    animal = str(animal)
    print(animal)
    prepare = PrepareGRIN2B(directory_path, animal)
    recording, brain_state_1, brain_state_2 = prepare.load_two_analysis_files( seizure = 'False')
    start_time_1, start_time_2, end_time_1, end_time_2 = prepare.get_start_end_times(start_times_GRIN2B_dict = start_time_GRIN2B_baseline, end_times_GRIN2B_dict = end_time_GRIN2B_baseline)
    data_1 = recording[:, start_time_1: end_time_1 + 1]
    data_2 = recording[:, start_time_2: end_time_2 + 1]
    print('all files loaded')
    explor_test_1 = FindNoiseThreshold(data = data_1, num_epochs = len(brain_state_1), brain_state_file = brain_state_1,
                                     noise_limit = 3000, channelvariables = channelvariables)
    explor_test_2 = FindNoiseThreshold(data = data_2, num_epochs = len(brain_state_2), brain_state_file = brain_state_2,
                                     noise_limit = 3000, channelvariables = channelvariables)
    clean_br_1, packet_loss_idx_1 = explor_test_1.find_packetloss_indices()
    slope_thresh_1, int_thresh_1 = explor_test_1.calc_noise_thresh(packet_loss_idx_1)
    noise_filter = NoiseFilter(data_1, num_epochs = len(brain_state_1), brain_state_file = brain_state_1, channelvariables = channelvariables,ch_type = 'eeg')
    bandpass_filtered_data_1 = noise_filter.filter_data_type()
    power_calc_full_rec_1, noise_indices_1 = noise_filter.power_calc_noise(bandpass_filtered_data_1, slope_thresh_1, int_thresh_1, clean_br = clean_br_1, br_number = 1)
    clean_br_2, packet_loss_idx_2 = explor_test_2.find_packetloss_indices()
    slope_thresh_2, int_thresh_2 = explor_test_2.calc_noise_thresh(packet_loss_idx_2)
    noise_filter = NoiseFilter(data_2, num_epochs = len(brain_state_2), brain_state_file = brain_state_2, channelvariables = channelvariables,ch_type = 'eeg')
    bandpass_filtered_data_2 = noise_filter.filter_data_type()
    power_calc_full_rec_2, noise_indices_2 = noise_filter.power_calc_noise(bandpass_filtered_data_2, slope_thresh_2, int_thresh_2, clean_br = clean_br_2, br_number = 1)
    br_array_1 = coherence_arrays(br_state_num = 2, br_state = clean_br_1, noise_indices = noise_indices_1, bandpass_filtered_data = bandpass_filtered_data_1)
    br_array_2 = coherence_arrays(br_state_num = 2, br_state = clean_br_2, noise_indices = noise_indices_2, bandpass_filtered_data = bandpass_filtered_data_2)
    os.chdir(coherence_save_path)
    np.save(str(animal) + '_nrem_br_1.npy', br_array_1)
    np.save(str(animal) + '_nrem_br_2.npy', br_array_2)
    print('dataframes for ' + str(animal) + ' saved')
    