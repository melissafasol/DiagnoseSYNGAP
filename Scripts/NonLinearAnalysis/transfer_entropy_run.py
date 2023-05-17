import os 
import sys
import numpy as np 
import pandas as pd 

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs


from transfer_entropy_functions import transfer_entropy

def channel_pairs(channels):
    pairs = []
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            pairs.append([channels[i], channels[j]])
    
    return pairs


directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
save_path = '/home/melissa/RESULTS/XGBoost/TransEnt'
channels = [0,1, 2,3,4,5,6,7,8,9,10,11,12,13]
channel_comb = channel_pairs(channels)

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
    filter_1 = np.moveaxis(np.array(np.split(bandpass_filtered_data_1, 17280, axis = 1)), 1, 0)
    filter_2 = np.moveaxis(np.array(np.split(bandpass_filtered_data_2, 17280, axis = 1)), 1, 0)
    epochs_1 = []
    epochs_2 = []
    for i in np.arange(0, 17280, 1):
        epoch_chans_1 = []
        epoch_chans_2 = []
        for i, channel in enumerate(channel_comb):
            print(i)
            chan_x = channel[0]
            chan_y = channel[1]
            ent_1 = transfer_entropy( filter_1[chan_x, i], filter_1[chan_y, i])
            ent_2 = transfer_entropy( filter_2[chan_x, i], filter_2[chan_y, i])
            epoch_chans_1.append(ent_1)
            epoch_chans_2.append(ent_2)
        epochs_1.append(epoch_chans_1)
        epochs_2.append(epoch_chans_2)
    trans_ent_1 = np.array(epochs_1)
    trans_ent_2 = np.array(epochs_2)
    os.chdir(save_path)
    np.save(animal + '_TransEnt_1.npy', trans_ent_1)
    np.save(animal + '_TransEnt_2.npy', trans_ent_2)