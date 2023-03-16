'''Script to save coherence arrays in the correct format for running on Jorge's GUI'''

import os 
import sys
import numpy as np
import pandas as pd


sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
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
    
    
def reshape_coherence_array(epochs_array):
    
    arr = np.moveaxis(epochs_array, 1, -1)
    br_arr = arr.reshape(-1, *arr.shape[-1:]).T
    
    return br_arr

