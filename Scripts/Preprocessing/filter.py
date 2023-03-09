import os 
import numpy as np 
import pandas as pd 
from scipy import signal 

from parameters import channel_numbers, channel_types


class NoiseFilter:
    
    order = 3
    sampling_rate = 250.4 
    nyquist = sampling_rate/2
    low = 0.2/nyquist
    high = 100/nyquist
    
    def __init__(self, unfiltered_data, number_of_epochs, noise_limit, brain_state_file, channel_type):
        '''input unfiltered data and the number of epochs data needs to be split into
        - this should be equivalent to the length of a brainstate file. Class will output new brainstate file
        with identified noisy epochs:
        unfiltered_data: unfiltered numpy array 
        number of epochs: number of bins to split data into 
        noise_limit: threshold (mV) beyond which data is labelled as noise 
        brain_state_file: file with all of the sleep scores
        channel_type: eeg, emg, all - if eeg only eeg channels selected, if emg only emg channels, if all both emg and eeg'''
        self.unfiltered_data = unfiltered_data
        self.number_of_epochs = number_of_epochs
        self.noise_limit = noise_limit
        self.brain_state_file = brain_state_file
        self.channel_type = channel_type
        
    def find_packetloss_indices(self):
        
        def packet_loss(epoch):
            mask = epoch.max() < self.noise_limit
            return mask 
        
        def get_dataset(data):
            packet_loss_score = []
            for epoch in data:
                packet_loss_score.append(0) if packet_loss(epoch) == True else packet_loss_score.append(6)
            return packet_loss_score
        
        def butter_bandpass(self):
            butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype = 'band', analog = False)
            filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
            return filtered_data
        
        def filter_data_type(channel_type):
            indices = []
            if channel_type == 'eeg':
                for idx, ch_type in enumerate(channel_types):
                    if ch_type == 'eeg':
                        indices.append(idx)
            if channel_type == 'emg':
                for idx, ch_type in enumerate(channel_types):
                    if ch_type == 'emg':
                        indices.append(idx)
            if channel_type == 'all':
                indices = channel_numbers
            return indices
        
        #label noise above 3000mV
        split_epochs = np.split(self.unfiltered_data, self.number_of_epochs, axis = 1) 
        packet_loss_score = get_dataset(split_epochs) 
        noise_indices = []
        for idx, i in enumerate(packet_loss_score):
            if i == 6:
                noise_indices.append(idx)
                
        #change identified noisy indices in brain state file 
        self.brain_state_file.loc[noise_indices, 'brainstate'] = 6
        
        #select all channels, only emg channels or only eeg channels 
        indices = filter_data_type(self.channel_type)
        selected_channels = self.unfiltered_data[indices, :]
        
        #bandpass filter data       
        bandpass_filtered_data = butter_bandpass(selected_channels)
        
                
        return self.brain_state_file, bandpass_filtered_data
        



        
   