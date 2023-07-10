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


#extract array object to save
data, times = raw_edf[:, :]

#extract data as a numpy array to save
data, times = raw_edf[:]

filtered_data = raw_edf.filter(l_freq = 0.3, h_freq = 35)
filtered_data.plot(start=4000,duration=10)