import sys
import os 
import pandas as pd
import numpy as np
import matplotlib
import mne

#read human data 
os.chdir('/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data')
raw_edf = mne.io.read_raw_edf('P1 N1_(1).edf', eog = [0],
            exclude = ['F3:M2', 'C3:M2', 'O1:M2', 'Position', 'PLM1', 'PLM2',
                       'Snore', 'Flow', 'Effort', 'Thorax', 'Abdomen', 'SpO2', 'Pleth', 'Pulse',
                      'E1:M2', 'E2:M2'], preload=True)

filtered_data = raw_edf.filter(l_freq = 0.3, h_freq = 35)

#extract array object to save
data, times = filtered_data[:, :]

#split data into 30 second epochs 

#calculate divisible length 
#sampling_rate = 256, seconds per epoch = 30
def split_into_epochs(data, sampling_rate, num_seconds):
    
    data_points = sampling_rate*num_seconds #256*30
    new_len = len(data[1]) - len(data[1])%data_points
    split_data = data[:, 0:new_len]
    number_epochs = new_len/sampling_rate/num_seconds
    epochs = np.split(split_data, number_epochs, axis = 1)
    
    return epochs

epochs = split_into_epochs(data = data, sampling_rate = 256, num_seconds = 30)



def identify_noisy_epochs(split_epochs, num_channels, num_epochs):
    channels_idx = list(np.arange(0, num_channels))
    epochs_idx = list(np.arange(0, num_epochs))
    
    intercept_noise = []
    slope_noise = []
    for chan in channels_ind:
        for epoch in epochs_ind:
            freq, power = scipy.signal.welch(test_epochs[epoch][chan], window='hann', fs=256, nperseg=7680)
            slope, intercept = np.polyfit(freq, power, 1)
            if intercept < 1e-13:
                int_noise_dict = {'Intercept': [intercept], 'Epoch_IDX': [epoch], 'Channel': [chan]}
                int_noise_df = pd.DataFrame(data = int_noise_dict)
                intercept_noise.append(int_noise_df)
            elif slope > -1e-13:
                slope_noise_dict = {'Slope': [slope], 'Epoch_IDX': [epoch], 'Channel': [chan]}
                slope_noise_df = pd.DataFrame(data = slope_noise_dict)
                slope_noise.append(slope_noise_df)
            
    intercept_noise_concat = pd.concat(intercept_noise)
    slope_noise_concat = pd.concat(slope_noise)
    
    return intercept_noise_concat, slope_noise_concat