import os 
import numpy as np 
import pandas as pd 
from scipy import signal 


class NoiseFilter():
    
    def __init__(self, unfiltered_data, number_of_epochs, noise_limit, brain_state_file):
        '''input unfiltered data and the number of epochs data needs to be split into
        - this should be equivalent to the length of a brainstate file. Class will output new brainstate file
        with identified noisy epochs'''
        self.unfiltered_data = unfiltered_data
        self.number_of_epochs = number_of_epochs
        self.noise_limit = noise_limit
        self.brain_state_file = brain_state_file
        
    def find_packetloss_indices(self):
        def packet_loss(epoch):
            mask = epoch.max() < self.noise_limit
            return mask 
        
        def get_dataset(data):
            packet_loss_score = []
            for epoch in data:
                packet_loss_score.append(0) if packet_loss(epoch) == True else packet_loss_score.append(6)
            return packet_loss_score
    
        split_epochs = np.split(self.unfiltered_data, self.number_of_epochs, axis = 1)  #split raw data into epochs
        packet_loss_score = get_dataset(split_epochs) #find indices where value in epoch exceeds 3000mV
        noise_indices = []
        for idx, i in enumerate(packet_loss_score):
            if i == 6:
                noise_indices.append(idx)
                
        #change identified noisy indices in brain state file 
        self.brain_state_file.loc[noise_indices, 'brainstate'] = 6
                
        return self.brain_state_file
        

class BandPassFilter:

    order = 3
    sampling_rate = 250.4 
    nyquist = 125.2
    low = 0.2/nyquist
    high = 100/nyquist
    
    def __init__(self, unfiltered_data, order, sampling_rate, lower_bound, upper_bound):
        '''order = 3
        sampling_rate = 250.4 
        nyquist = 125.2
        low = 0.2/nyquist
        high = 100/nyquist'''
        
        self.unfiltered_data = unfiltered_data
        self.order = order
        self.sampling_rate = sampling_rate
        self.nyquist = sampling_rate/2
        self.low = lower_bound/self.nyquist
        self.high = upper_bound/self.nyquist
        

    def butter_bandpass(self):
        #stripped filter function to apply bandpass filter to entire recording before time and frequency domain calculations
        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype = 'band', analog = False)
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
        return filtered_data
