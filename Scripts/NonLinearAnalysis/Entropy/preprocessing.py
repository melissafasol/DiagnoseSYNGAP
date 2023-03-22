import os 
import sys
import numpy as np 
import pandas as pd 
from scipy import signal

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from filter import NoiseFilter


class EntropyFilter(NoiseFilter):
    '''Class inherits from filtering class in Preprocessing folder but changes lower and upper
    bound of IIR filter for entropy analysis to avoid power-line artifacts'''
    def __init__(self, unfiltered_data, brain_state_file, channelvariable, ch_type):
        super().__init__(unfiltered_data, brain_state_file, channelvariable, ch_type)
        self.order = 3
        self.low = 0.2/125.2
        self.high = 48/125.2
        
    def select_eeg_channels(self, data):
        eeg_indices = [0,2,3,4,5,6,7,8,9,10,11,12,13,15]
        eeg_data = data[eeg_indices, :]
        return eeg_data
    
    def prepare_data_files(self, clean_br, eeg_data, br_number):
        
        br_indices = clean_br.loc[clean_br['brainstate'] == br_number].index.tolist()
        data_split = np.split(eeg_data, len(clean_br), axis = 1) 
        
        return br_indices, data_split