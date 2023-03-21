import os 
import sys
import numpy as np 
import pandas as pd 
from scipy import signal

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from filter import NoiseFilter


class EntropyFilter(NoiseFilter):
    
    def __init__(self, unfiltered_data, brain_state_file, channelvariable, ch_type):
        super.__init__(unfiltered_data, brain_state_file, channelvariable, ch_type)
        self.low = 0.2/125.2
        self.high = 48/125.2
    
    def entropy_bandpass_filter(self):
        
        def butter_bandpass(data):
            butter_b, butter_a = signal.butter(3, [0.2/125.2, 48/125.2], btype= 'band', analog=False)
            filtered_data = signal.filtfilt(butter_b, butter_a, data)
            return filtered_data
        
        indices = []
        if self.ch_type == 'eeg':
            for idx, ch in enumerate(self.channel_types):
                if ch == 'eeg':
                        indices.append(idx)
            if self.ch_type == 'emg':
                for idx, ch in enumerate(self.channel_types):
                    if ch == 'emg':
                        indices.append(idx)
            if self.ch_type == 'all':
                indices = self.channel_numbers
        
        #Select all, emg, or eeg channel indices to apply bandpass filter                                    
        selected_channels = self.unfiltered_data[indices, :]     
        bandpass_filtered_data=butter_bandpass(data=selected_channels) 
                        
        return bandpass_filtered_data