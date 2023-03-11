import os 
import numpy as np 
import pandas as pd 
import scipy
from scipy import signal 

#from parameters import channelvariables


class NoiseFilter:
    """Class to input unfiltered data, returns bandpass filtered data and calculates whether epoch
    is within specified noise threshold.
    """
    
    order = 3
    sampling_rate = 250.4 
    nyquist = sampling_rate/2
    low = 0.2/nyquist
    high = 100/nyquist

    
    def __init__(self, unfiltered_data, num_epochs, brain_state_file, channelvariables,ch_type):
        self.unfiltered_data = unfiltered_data 
        self.num_epochs = num_epochs                        #number of epochs to split raw data into 
        self.brain_state_file = brain_state_file                        #dataframe with brainstates  
        self.channel_variables = channelvariables                      #dictionary with channel types and channel variables 
        self.channel_types= channelvariables['channel_types']
        self.channel_numbers = channelvariables['channel_numbers']
        self.ch_type = ch_type                                          #specify channel type to perform calculations on
        
        
        def butter_bandpass(data):
            butter_b, butter_a = signal.butter(self.order, [self.low, self.high], btype = 'band', analog = False)
            filtered_data = signal.filtfilt(butter_b, butter_a, data)
            return filtered_data

        def filter_data_type():
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
            return indices
        
        #Select all, emg, or eeg channel indices to apply bandpass filter 
        indices = filter_data_type()                                    
        selected_channels = self.unfiltered_data[indices, :]     
        bandpass_filtered_data=butter_bandpass(data=selected_channels) 
                        
        return bandpass_filtered_data

    
    def power_calc_noise(self, bandpass_filtered_data, slope_thresh, int_thresh):
        
        def lin_reg_calc(epoch):
            noise_array = []
            power_array = []

            freq, power = scipy.signal.welch(epoch, window = 'hann', fs = 250.4, nperseg = 1252)
            slope, intercept = np.polyfit(freq, power, 1)
            power_array.append(power)
            if intercept > int_thresh or slope < slope_thresh:
                noise_array.append(5)
            else:
                noise_array.append(0)
            
            return noise_array, power_array    
        
        def apply_lin_reg(bandpass_filtered_data, clean_br, br_number):
            '''function applies lin_reg_calc function to entire time series and returns two arrays,
            one with noise labels and one with power calculation results'''
            split_epochs = np.split(bandpass_filtered_data, self.number_of_epochs, axis = 1)
            noise_per_epoch = []
            power_calc = []
            packet_loss = (clean_br.query('brainstate == 6')).index.tolist()
            br_calc = (clean_br.query('brainstate == str(br_number)'))
            for idx, epoch in enumerate(split_epochs):
                if idx in packet_loss:
                    pass
                else:
                    channel_arrays = []
                    for chan in epoch:
                        power= lin_reg_calc(chan)
                        channel_arrays.append(power)
                        one_epoch_arrays = np.dstack(channel_arrays)
                        one_epoch_power = np.vstack(one_epoch_arrays[1][0])
                        power_calc.append(one_epoch_power)
                        noise_per_epoch.append(one_epoch_arrays[0][0].T)
            
            power_array = np.array(power_calc)
            label_array = np.array(noise_per_epoch)
            return power_array, label_array
        
        power_array, label_array = apply_lin_reg(bandpass_filtered_data)
        return power_array, label_array
            