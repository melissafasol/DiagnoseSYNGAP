'''script to choose noise parameter values for filtering'''

import os 
import numpy as np 
import pandas as pd 
import scipy 
import sys


class FindNoiseThreshold:
    
    """Class to calculate the average gradient and intercept of the spectral slope.
    
    These variables will be used to determine the threshold for noise in the Filter class.
    """
    
    def __init__(self, data,  brain_state_file, noise_limit, channelvariables, br_number):
        self.data=data 
        self.brain_state_file = brain_state_file
        self.num_epochs = len(brain_state_file)
        self.br_number = br_number
        self.noise_limit = noise_limit          #threshold (mV) beyond which data is labelled as noise
        self.ch_variables = channelvariables        #dictionary with channel types and channel variables 
        self.channel_types= channelvariables['channel_types']
        self.channel_numbers = channelvariables['channel_numbers']
    
    #Label epochs above noise threshold
    def find_packetloss_indices(self):   
        
        def packet_loss(epoch):
            mask = epoch.max() < self.noise_limit
            return mask 
        
        def get_dataset(data):
            packet_loss_score = []
            for epoch in data:
                packet_loss_score.append(0) if packet_loss(epoch) == True else packet_loss_score.append(6) 
            return packet_loss_score
        
        split_epochs = np.split(self.data, self.num_epochs, axis = 1) 
        packet_loss_score = get_dataset(split_epochs) 
        noise_indices = []
        for idx, i in enumerate(packet_loss_score):
            if i == 6:
                noise_indices.append(idx)
                
        #Change identified noisy indices in brain state file 
        self.brain_state_file.loc[noise_indices, 'brainstate'] = 6
        
        return self.brain_state_file, noise_indices
    
    def calc_noise_thresh(self, noise_indices):
        '''This function calculates the mean and standard dev which calculates
        the threshold over which to label an epoch as noisy or clean per brainstate
        '''
        def average_slope_intercept(epoch):
            freq, power = scipy.signal.welch(epoch, window='hann', fs=250.4, nperseg=1252)
            slope, intercept = np.polyfit(freq, power, 1)
            return slope, intercept
        
        slope_ls = []
        intercept_ls = []
        split_data = np.split(self.data, self.num_epochs, axis=1)
        br_number_indices = self.brain_state_file.loc[self.brain_state_file['brainstate'] == self.br_number].index.tolist()
        for idx, epoch in enumerate(split_data):
            if idx in noise_indices:
                pass
            elif idx in br_number_indices:
                slope_int_res = [average_slope_intercept(chan) for chan in epoch]
                slope_int_arr = np.array(slope_int_res).T
                slope_ls.append(slope_int_arr[0])
                intercept_ls.append(slope_int_arr[1])
            else:
                pass
        
        all_values_slope = np.array(slope_ls)
        all_values_intcpt = np.array(intercept_ls)
        slope_mean = np.mean((np.array(slope_ls)), axis=0)
        int_mean = np.mean((np.array(intercept_ls)), axis=0)
        std_slope = np.std(slope_mean, axis = 0)
        std_int = np.std(int_mean, axis = 0)
        
        std_int_max = int(std_int.max())
        int_thresh = int(int_mean.max()) + 3*std_int_max
        
        slope_thresh = round(slope_mean.min() - 3*std_slope)
        
        return  slope_thresh, int_thresh