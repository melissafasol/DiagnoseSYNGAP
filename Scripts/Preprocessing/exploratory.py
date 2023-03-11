'''script to choose noise parameter values for filtering'''

import os 
import numpy as np 
import pandas as pd 
import scipy 
import sys


class FindNoiseThreshold:
    
    def __init__(self, packet_loss_indices, data, number_of_epochs):
        self.packet_loss_indices = packet_loss_indices
        self.data = data 
        self.number_of_epochs = number_of_epochs
    
    def calc_avg_slope(self):
        
        def average_slope_intercept(epoch):
            freq, power = scipy.signal.welch(epoch, window = 'hann', fs = 250.4, nperseg = 1252)
            slope, intercept = np.polyfit(freq, power, 1)
            return slope, intercept
        
        slope_ls = []
        intercept_ls = []
        split_data = np.split(self.data, self.number_of_epochs, axis = 1)
        for idx, epoch in enumerate(split_data):
            if idx in self.packet_loss_indices:
                pass
            else:
                slope_int_res = [average_slope_intercept(chan) for chan in epoch]
                slope_int_arr = np.array(slope_int_res).T
                slope_ls.append(slope_int_arr[0])
                intercept_ls.append(slope_int_arr[1])
        
        slope_mean = np.mean((np.array(slope_ls)), axis = 0)
        int_mean = np.mean((np.array(intercept_ls)), axis = 0)
        std_slope = np.std(slope_mean, axis = 0)
        std_int = np.std(int_mean, axis = 0)
        
        return slope_mean, int_mean, std_slope, std_int
