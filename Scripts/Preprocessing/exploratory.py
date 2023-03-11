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
    
    def __init__(self, noise_idx, data, num_epochs):
        self.noise_idx=noise_idx
        self.data=data 
        self.num_epochs=num_epochs
    
    
    def calc_avg_slope(self):
        
        def average_slope_intercept(epoch):
            freq, power = scipy.signal.welch(epoch, window='hann', fs=250.4, nperseg=1252)
            slope, intercept = np.polyfit(freq, power, 1)
            return slope, intercept
        
        slope_ls = []
        intercept_ls = []
        split_data = np.split(self.data, self.number_of_epochs, axis=1)
        for idx, epoch in enumerate(split_data):
            if idx in self.noise_idx:
                pass
            else:
                slope_int_res = [average_slope_intercept(chan) for chan in epoch]
                slope_int_arr = np.array(slope_int_res).T
                slope_ls.append(slope_int_arr[0])
                intercept_ls.append(slope_int_arr[1])
        
        slope_mean = np.mean((np.array(slope_ls)), axis=0)
        int_mean = np.mean((np.array(intercept_ls)), axis=0)
        std_slope = np.std(slope_mean, axis = 0)
        std_int = np.std(int_mean, axis = 0)
        
        return slope_mean, int_mean, std_slope, std_int

    
    def calc_noise_threshold(self, slope_mean, int_mean, std_slope, std_int):
        
        '''This function takes the slope mean for each channel and calculates
        the mean and standard dev which calculates the threshold over which
        to label an epoch as noisy or clean.
        '''
        std_int_max = int(std_int.max())
        int_thresh = int(int_mean.max()) + 3*std_int_max
        
        slope_thresh = round(slope_mean.min() - 3*std_slope)
        
        return slope_thresh, int_thresh