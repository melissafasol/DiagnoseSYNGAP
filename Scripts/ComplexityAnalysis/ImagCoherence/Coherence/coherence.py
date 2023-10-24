'''Script to save coherence arrays in the correct format for running on Jorge's GUI'''

import os 
import sys
import numpy as np
import pandas as pd


sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from exploratory import FindNoiseThreshold
from filter import NoiseFilter
from constants import channel_variables


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
    
    
def reshape_coherence_array(epochs_array):
    
    arr = np.moveaxis(epochs_array, 1, -1)
    br_arr = arr.reshape(-1, *arr.shape[-1:]).T
    
    return br_arr

#def hardcode_channel_combinations():
    
    #OG
    #comb_short_distance = [(0,1), (0,2,)....]
    #len_short = 30
    #comb_long_distance = [(0,7), (0,8)....]
    #len_long = 50
    
short_intrahemispheric = [(0, 4), (0,6), (1,2), (1,3), (2,3), (3,6),
                              (4,5), (7,8), (8,13), (9,10), (9,13), (10,11),
                              (10,12), (11,12)]
    

long_interhemispheric =  [ (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0,13),
                          (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1,13),
                          (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2,13),
                          (3, 7), (3, 8), (3, 9), (3, 10),(3, 11), (3, 12), (3,13),
                          (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4,13),
                          (5, 7), (5, 8),(5, 9), (5, 10), (5, 11), (5, 12), (5,13), 
                          (6, 7), (6, 8),(6, 9), (6, 10), (6, 11), (6, 12), (6,13)]


