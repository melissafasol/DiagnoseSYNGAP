import numpy as np
import pandas as pd
from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp
import EntropyHub as EH

class DispersionEntropy:
    
    def __init__(self, dat_array):
        self.dat_array = dat_array
        
        
    def disp_en(self):
        disp_en_ls =  []

        for epoch in self.dat_array:
            Dispx_1, Ppi_1 = EH.DispEn(epoch, m = 3, tau = 2, c = 4, Typex = 'ncdf')
            disp_en_ls.append(Dispx_1)
        
        disp_en_array = np.array(disp_en_ls)
        
        return disp_en_array

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

    def transform_to_dataframe(self, complexity_array, channel_labels, complexity_value):
        num_channels, num_epochs = complexity_array.shape
        epochs = np.tile(np.arange(num_epochs), num_channels)
        channels = np.repeat(channel_labels, num_epochs)
        complex_values = complexity_array.flatten()

        df = pd.DataFrame({
        'Epoch': epochs,
        'Channel': channels,
        f'{complexity_value}': complex_values
        })

        return df
