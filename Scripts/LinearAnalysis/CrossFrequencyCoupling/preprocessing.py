'''Script to filter gamma/theta band and identify troughs using the z-score moving mean algorithm'''


import os 
import sys
import pandas as pd
import numpy as np
from scipy import signal, linalg
from sklearn.metrics import r2_score
from neurodsp.utils import create_times





class FilterCFC():
    
    order = 3
    sampling_rate = 250.4
    nyquist = 125.2
    
    def __init__(self, unfiltered_data, filter_type):
        self.unfiltered_data = unfiltered_data
        self.filter_type = filter_type
        if self.filter_type == 'gamma':
            self.low = 30/self.nyquist
            self.high = 48/self.nyquist
        elif self.filter_type == 'theta':
            self.low = 5/self.nyquist
            self.high = 10/self.nyquist
    
    def butter_bandpass(self):
        butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype='band', analog = False)
        
        filtered_data = signal.filtfilt(butter_b, butter_a, self.unfiltered_data)
        
        return filtered_data
    
class ExtractThetaTroughIndices():
    
    sampling_rate = 250.4

    def __init__(self, theta_filtered, start_time_seiz, end_time_seiz, length_theta_trough):
        '''theta_filtered = input filtered np array of data filtered with bandpass filter for theta band (all channels),
        start_time_seiz = start time (seconds*sampling_rate) of epoch of interest (e.g seizure)
        start_time_seiz = end time (seconds*sampling_rate) of epoch of interest
        length_theta_trough = if looking at 10 seconds directly before seizure this is >> (20, 30)'''
        self.theta_filtered = theta_filtered 
        self.theta_amp = np.abs(signal.hilbert(self.theta_filtered)) #extract amplitude of theta band 
        self.start_time = start_time_seiz
        self.end_time = end_time_seiz
        self.length = (self.end_time - self.start_time)/self.sampling_rate #length of seizure array
        self.times = create_times(self.length, self.sampling_rate) #creates time array
        self.length_theta_trough = (20, 30)
        self.tidx = np.logical_and(self.times >= self.length_theta_trough[0], self.times < length_theta_trough[1])
    
    def apply_moving_mean(self, channel):
        #channel to calculate peaks and troughs 
        def thresholding_algo( y, lag, threshold, influence):
            signals = np.zeros(len(y))
            filteredY = np.array(y)
            avgFilter = [0]*len(y)
            stdFilter = [0]*len(y)
            avgFilter[lag - 1] = np.mean(y[0:lag])
            stdFilter[lag - 1] = np.std(y[0:lag])
            for i in range(lag, len(y)):
                if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                    if y[i] > avgFilter[i-1]:
                        signals[i] = 1
                    else:
                        signals[i] = -1

                    filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                    avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                    stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
                else:
                    signals[i] = 0
                    filteredY[i] = y[i]
                    avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                    stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

            return np.asarray(signals),np.asarray(avgFilter),np.asarray(stdFilter)

        #for only one channel
        signals, avgfilter, stdfilter = thresholding_algo(y = self.theta_amp[channel, self.tidx], lag = 5, threshold = 4, influence = 0.75) 
        
        return signals
    
    
    def extract_trough_indices(self, signals):
        trough_indices = []
        pre_trough_indices = []
        post_trough_indices = []
        
        for idx, i in enumerate(signals):
            if signals[idx] == -1. and signals[idx + 1] == -1.:
                print('two troughs idx')
                trough_indices.append(idx)
                continue
            elif signals[idx] == -1. and signals[idx - 1] == 0. and signals[idx + 1] == 0.:
                print('only one trough idx')
                trough_indices.append(idx)
            else:
                pass
        
        for i in trough_indices:
            pre_trough_indices.append([i - 26, i-1])
            post_trough_indices.append([i + 1, i + 26])
        
        return trough_indices, pre_trough_indices, post_trough_indices
    