import sys
import os 
import pandas as pd
import numpy as np
#%matplotlib qt
import mne
#%matplotlib inline

import matplotlib.pyplot as plt
#matplotlib.use('agg')

from load_files import LoadFiles
#from exploratory import 
from filter import NoiseFilter
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables

#load an example file 
directory_path = '/home/melissa/PREPROCESSING/SYNGAP1/numpyformat_baseline/'
tst_72 = np.load(directory_path + 'S7072_BASELINE.npy')
start_time = 16481329
end_time = 38115888

number_of_channels = 16
sample_rate = 250.4
sample_datatype = 'int16'
display_decimation = 1

ch_names = ['S1Tr_RIGHT', 'EMG_RIGHT', 'M2_FrA_RIGHT','M2_ant_RIGHT','M1_ant_RIGHT', 'V2ML_RIGHT',
            'V1M_RIGHT', 'S1HL_S1FL_RIGHT', 'V1M_LEFT', 'V2ML_LEFT', 'S1HL_S1FL_LEFT',
            'M1_ant_LEFT','M2_ant_LEFT','M2_FrA_LEFT', 'EMG_LEFT', 'S1Tr_LEFT']

ch_types = ['eeg', 'emg', 'eeg', 'eeg', 'eeg', 'eeg',
           'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
           'eeg', 'eeg', 'eeg', 'emg', 'eeg']

channel_colors = {'eeg': 'purple'}


raw_info = mne.create_info(ch_names, sfreq = 250.4, ch_types=ch_types)

train_2_ids = ['S7072']
for animal in train_2_ids:
    animal = str(animal)
    print(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = SYNGAP_baseline_start, end_times_dict = SYNGAP_baseline_end)
    noise_filter_1 = NoiseFilter(data_1, brain_state_file = brain_state_1, channelvariables = channel_variables,ch_type = 'all')
    noise_filter_2 = NoiseFilter(data_2, brain_state_file = brain_state_2 , channelvariables = channel_variables,ch_type = 'all')
    bandpass_filtered_data_1 = noise_filter_1.filter_data_type()
    bandpass_filtered_data_2 = noise_filter_2.filter_data_type()
    
    
clean_indices = pd.read_pickle('/home/melissa/PREPROCESSING/SYNGAP1/cleaned_br_files/' + 'S7072_BL1.pkl')
noise_indices = clean_indices.loc[clean_indices['brainstate'] == 6]
indices = list(noise_indices.index)

#find indices to plot
br_1_72 = pd.read_csv('/home/melissa/RESULTS/ICASSP/ALL_EPOCHS/Int_Slope/' + 'S7072_br_1_all_epochs.csv') 
noisy_epochs = br_1_72[(br_1_72['Slope'] < -10)]
unique_epochs = np.unique(noisy_epochs['Epoch'])
len(unique_epochs)

noise_indices = []
for idx in unique_epochs:
    if idx in indices:
        pass
    else:
        noise_indices.append(idx)
        
split_epochs = np.split(bandpass_filtered_data_1, 17280, axis = 1)

epoch_test = bandpass_filtered_data_1[:, start_time:]

raw = mne.io.RawArray(epoch_test, raw_info)
raw.plot(scalings = 'auto', start = 0, show_scrollbars = False)
plt.show()