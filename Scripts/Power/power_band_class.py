import os 
import sys
import pandas as pd 
import numpy as np
import scipy

directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline'
results_path = '/home/melissa/RESULTS/XGBoost/SYNGAP1/'
 
motor = [1,2,3,10,11,12]
visual = [4, 5, 7, 8]
somatosensory = [0, 6, 9, 13]


class PowerBands():
    
    
    def __init__(self, freq_low, freq_high, fs, nperseg):
        '''
        freq_low = lower frequency bound e.g 2Hz
        freq_high = higher frequency bound e.g 10Hz
        animal_ls = ['S7070', 'S7071']
        two_recordings = if false then animal recordings only have one recording but if true then have two
        
        '''
        self.freq_low = int(freq_low/0.2)
        self.freq_high = int(freq_high/0.2 + 1)
        self.fs= fs
        self.nperseg = nperseg
        
    def power_band_calculations(self, data_array, clean_indices):

        band_power_ls = []

        for idx, epoch in enumerate(data_array):
            if idx not in clean_indices:
                continue
            freq, power = scipy.signal.welch(epoch, window='hann', fs = self.fs, nperseg = self.nperseg)
            freq_power = np.mean(power[self.freq_low:self.freq_high])
            band_power_ls.append(freq_power)
    
        power_array = np.array(band_power_ls)
        
        return power_array
    
    def functional_region_power_recording(self, bandpass_filtered_data, motor_indices, visual_indices, somatosensory_indices, clean_indices):
        
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
            motor_power_calc = self.power_band_calculations(motor_filtered_channel_data, clean_indices)
            motor_power_ls.append(motor_power_calc)
    
        # Process data for visual region
        for visual_channel in visual_indices:
            visual_filtered_channel_data = np.split(bandpass_filtered_data[visual_channel], 17280, axis=0)
            visual_power_calc = self.power_band_calculations(visual_filtered_channel_data, clean_indices)
            visual_power_ls.append(visual_power_calc)
            
        # Process data for somatosensory region
        for soma_channel in somatosensory_indices:
            somato_filtered_channel_data = np.split(bandpass_filtered_data[soma_channel], 17280, axis=0)
            somatosensory_power_calc = self.power_band_calculations(somato_filtered_channel_data, clean_indices)
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
    
    
    def calculate_power_average(self, bandpass_filtered_data, clean_indices, br_values, animal, frequency, all_channels = True, channel = None):
        'function to calculate power per channel'

        if not all_channels and channel is not None:
            channel_power = []
            split_data = np.array_split(bandpass_filtered_data[channel], indices_or_sections=int(bandpass_filtered_data[channel].shape[0]/1252), axis=0)
            epoch_num = 0
            for epoch in split_data:
                if len(epoch) != 1252:  # Ensure each split/epoch has the intended number of samples
                    continue
                if epoch_num not in clean_indices:
                    epoch_num += 1
                    continue
                freq, power = scipy.signal.welch(epoch, window='hann', fs=self.fs, nperseg=self.nperseg)
                freq_power = np.mean(power[self.freq_low:self.freq_high])
                channel_power.append(freq_power)
                epoch_num += 1
            one_channel_power_df = pd.DataFrame({'Power': channel_power,'Channel': [channel]*len(channel_power), 'Idx': clean_indices,
                                            'Brainstate': br_values, 'Animal_ID': [animal]*len(channel_power), 'Frequency_Band': [frequency]*len(channel_power) })

            return one_channel_power_df

        if not all_channels and channel is None:
            return 'Error: enter a channel number or set to process all channels'
        else:

            channels = list(range(bandpass_filtered_data.shape[0]))
            power_all_channels = []

            for channel_idx in channels:
                print(channel_idx)
                channel_power = []
                #split data into epochs to only select clean epoch indices
                split_data = np.array_split(bandpass_filtered_data[channel_idx], indices_or_sections=int(bandpass_filtered_data[channel_idx].shape[0]/1252), axis=0)
                #keep track of epoch number to add idx column to dataframe
                epoch_num = 0
                for epoch in split_data:
                    if len(epoch) != 1252:  # Ensure each split/epoch has the intended number of samples
                        continue
                    if epoch_num not in clean_indices:
                        epoch_num += 1
                        continue
                    freq, power = scipy.signal.welch(epoch, window='hann', fs=self.fs, nperseg=self.nperseg)
                    freq_power = np.mean(power[self.freq_low:self.freq_high])
                    channel_power.append(freq_power)
                    epoch_num += 1
                channel_power_df = pd.DataFrame({'Power': freq_power, 'Channel': [channel_idx]*len(channel_power), 'Idx': clean_indices,
                                            'Brainstate': br_values, 'Animal_ID': [animal]*len(channel_power), 'Frequency_Band': [frequency]*len(channel_power) })
                power_all_channels.append(channel_power_df)

            power_concat = pd.concat(power_all_channels, axis = 0)

            return power_concat
        
    def average_psd_overall(self, data_array, clean_indices): 
        power_array_ls = []
        split_data = np.array_split(data_array, 17280, axis=0)
        for idx, data in enumerate(split_data):
            if idx not in clean_indices:
                continue
            power_calculations = scipy.signal.welch(data, window = 'hann', fs = self.fs, nperseg = self.nperseg)
            frequency = power_calculations[0]
            power_array_ls.append(power_calculations[1])
            
        df_psd = pd.DataFrame(power_array_ls)
        mean_values = df_psd.mean(axis = 0)
        mean_psd = mean_values.to_numpy()
            
        return mean_psd, frequency
    
    def power_band_human(self, epoch):

        freq, power = scipy.signal.welch(epoch, window='hann', fs = self.fs, nperseg = self.nperseg)
        
        return freq, power
    
    