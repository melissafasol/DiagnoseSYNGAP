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



