import numpy as np
import pandas as pd
from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

class ComplexClass:
    
    def __init__(self, complexity_calculation, channel_labels):
        self.complexity_calculation = complexity_calculation
        self.channel_labels = channel_labels
        self.num_channels = len(channel_labels)
    
    def process_recordings(self, bandpass_filtered_data, concatenated=False):
        complexity_values = []
        split_size = 34560 if concatenated else 17280
        
        for idx in range(self.num_channels):
            filter_data = np.split(bandpass_filtered_data[idx], split_size, axis=0)
            if self.complexity_calculation == 'hurst':
                complexity_values.append(self.calculate_hurst_exp(filter_data))
            elif self.complexity_calculation == 'hfd':
                complexity_values.append(self.calculate_hfd_exp(filter_data))
            else:
                raise ValueError("Invalid complexity measure")
        
        complexity_array = np.array(complexity_values)
        return complexity_array

    def calculate_hurst_exp(self, channel_data):
        hurst_values = [compute_hurst_exp(np.expand_dims(epoch, axis=0)) for epoch in channel_data]
        return np.array(hurst_values).flatten()

    def calculate_hfd_exp(self, channel_data):
        hfd_values = [compute_higuchi_fd(np.expand_dims(epoch, axis=0)) for epoch in channel_data]
        return np.array(hfd_values).flatten()