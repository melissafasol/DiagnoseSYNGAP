import os 
import sys
import pandas as pd 
import numpy as np
import scipy


directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/'

SYNGAP_2_ID_ls =  ['S7086', 'S7096', 'S7070','S7072','S7083', 'S7063','S7064','S7069', 'S7086','S7091'] 
SYNGAP_1_ID_ls =  ['S7076', 'S7101', 'S7088', 'S7092', 'S7094' , 'S7098', 'S7068', 'S7074', 'S7076', 'S7071', 'S7075']
 
motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]
somatosensory = [0, 6, 9, 13]


class PowerBands():
    
    
    def __init__(self, data_array, freq_low, freq_high, fs, nperseg):
        '''
        freq_low = lower frequency bound e.g 2Hz
        freq_high = higher frequency bound e.g 10Hz
        animal_ls = ['S7070', 'S7071']
        two_recordings = if false then animal recordings only have one recording but if true then have two
        
        '''
        self.data_array = data_array
        self.freq_low = freq_low/0.2
        self.freq_high = freq_high/0.2
        self.fs= fs
        self.nperseg = nperseg
        
    def power_band_calculations(self):

        band_power_ls = []

        for epoch in self.data_array:
            freq, power = scipy.signal.welch(epoch, window='hann', fs = self.fs, nperseg = self.nperseg)
            freq_power = np.mean(power[self.freq_low:self.freq_high + 1])
            band_power_ls.append(freq_power)
    
        power_array = np.array(band_power_ls)
        
        return power_array
    
    def functional_region_power_recording(self, bandpass_filtered_data, motor_indices, visual_indices, somatosensory_indices):
        
        '''
        bandpass_filtered_data = filtered data array
        motor_indices = motor region channel indices
        visual_indices = visual region channel indices
        somatosensory_indices = somatosensory region channel indices
        '''
        
        motor_power_ls = []
        visual_power_ls = []
        somatosensory_power_ls = []

        # Process data for motor region
        for motor_channel in motor_indices:
            motor_filtered_channel_data = np.split(bandpass_filtered_data[motor_channel], 17280, axis=0)
            motor_power_calc = self.power_band_calculations(motor_filtered_channel_data)
            motor_power_ls.append(motor_power_calc)
    
        # Process data for visual region
        for visual_channel in visual_indices:
            visual_filtered_channel_data = np.split(bandpass_filtered_data[visual_channel], 17280, axis=0)
            visual_power_calc = self.power_band_calculations(visual_filtered_channel_data)
            visual_power_ls.append(visual_power_calc)
            
        # Process data for somatosensory region
        for soma_channel in somatosensory_indices:
            somato_filtered_channel_data = np.split(bandpass_filtered_data[soma_channel], 17280, axis=0)
            somatosensory_power_calc = self.power_band_calculations(somato_filtered_channel_data)
            somatosensory_power_ls.append(somatosensory_power_calc)
            
        # Combine the lists of motor power into NumPy arrays
        motor_power_arr = np.array(motor_power_ls)
        # Calculate means for specific channels
        motor_mean = np.mean(motor_power_arr, axis=0)

        # Combine the lists of motor power into NumPy arrays
        visual_power_arr = np.array(visual_power_ls)
        # Calculate means for specific channels
        visual_mean = np.mean((visual_power_arr), axis=0)

        # Combine the lists of motor power into NumPy arrays
        soma_power_arr = np.array(somatosensory_power_ls)
        # Calculate means for specific channels
        soma_mean = np.mean((soma_power_arr), axis=0)
        
        return motor_mean, visual_mean, soma_mean
    