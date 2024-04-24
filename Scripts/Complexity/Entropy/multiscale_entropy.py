import EntropyHub as EH
import random 
import os 
import sys
import numpy as np 
import pandas as pd
from scipy import average, gradient
import scipy
from scipy.fft import fft, fftfreq
from scipy import signal

from preprocessing import EntropyFilter

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import start_time_GRIN2B_baseline, end_time_GRIN2B_baseline, GRIN_het_IDs, GRIN2B_ID_list, GRIN2B_seiz_free_IDs, channel_variables, GRIN_wt_IDs


directory_path = '/home/melissa/PREPROCESSING/GRIN2B/GRIN2B_numpy'
eeg_indices = [0,2,3,4,5,6,7,8,9,10,11,12,13,15]
br_number = 2

for animal in GRIN2B_ID_list:
    print('loading ' + str(animal))
    animal = str(animal)
    load_files = LoadFiles(directory_path, animal)
    data_1, data_2, brain_state_1, brain_state_2 = load_files.load_two_analysis_files(start_times_dict = start_time_GRIN2B_baseline, end_times_dict = end_time_GRIN2B_baseline)
    #only select eeg channels and filter with bandpass butterworth filter before selecting indices
    preprocess_data_1 = EntropyFilter(unfiltered_data = data_1, brain_state_file = brain_state_1, channelvariable = channel_variables, ch_type = 'eeg')
    preprocess_data_2 = EntropyFilter(unfiltered_data = data_2, brain_state_file = brain_state_2, channelvariable = channel_variables, ch_type = 'eeg')
    filtered_data_1 = preprocess_data_1.filter_data_type()
    filtered_data_2 = preprocess_data_2.filter_data_type()
    br_indices_1, data_split_1 = preprocess_data_1.prepare_data_files(brain_state_1, filtered_data_1, br_number)
    br_indices_2, data_split_2 = preprocess_data_2.prepare_data_files(brain_state_2, filtered_data_2, br_number)
    Mobj = EH.MSobject('SampEn', m = 2, r = 0.5)
    Mobj.Func
    entropy_values = []
    for value in br_indices_1:
        epoch = data_split_1[value]
        entr_channel = []
        for chan_idx, chan in zip(eeg_indices, epoch):
            print('starting chan ' + str(chan_idx))
            MSx, Ci = EH.cMSEn(chan, Mobj, Scales = 5, Refined = True)
            Scale_1 = MSx[0]
            Scale_2 = MSx[1]
            Scale_3 = MSx[2]
            Scale_4 = MSx[3]
            Scale_5 = MSx[4]
            multiscale_dict = {'Animal_ID': [animal], 'Channel': [chan_idx], 'Scale_1': [Scale_1],
                          'Scale_2': [Scale_2], 'Scale_3': [Scale_3], 'Scale_4': [Scale_4],
                          'Scale_5': [Scale_5], 'Confidence_Interval': [Ci]}
            multiscale_data = pd.DataFrame(data = multiscale_dict)
            entr_channel.append(multiscale_data)
        chan_concat = pd.concat(entr_channel)
        entropy_values.append(chan_concat)
    for value in br_indices_2:
        epoch = data_split_2[value]
        entr_channel = []
        for chan_idx, chan in zip(eeg_indices, epoch):
            print('starting chan ' + str(chan_idx))
            MSx, Ci = EH.cMSEn(chan, Mobj, Scales = 5, Refined = True)
            Scale_1 = MSx[0]
            Scale_2 = MSx[1]
            Scale_3 = MSx[2]
            Scale_4 = MSx[3]
            Scale_5 = MSx[4]
            multiscale_dict = {'Animal_ID': [animal], 'Channel': [chan_idx], 'Scale_1': [Scale_1],
                          'Scale_2': [Scale_2], 'Scale_3': [Scale_3], 'Scale_4': [Scale_4],
                          'Scale_5': [Scale_5], 'Confidence_Interval': [Ci]}
            multiscale_data = pd.DataFrame(data = multiscale_dict)
            entr_channel.append(multiscale_data)
        chan_concat = pd.concat(entr_channel)
        entropy_values.append(chan_concat)
    entr_df = pd.concat(entropy_values)
    entr_df.to_csv('/home/melissa/RESULTS/GRIN2B/Entropy/MULTISCALE_SAMP/REM' + str(animal) + '_ms_entr_' + str(br_number) + '.csv')
    print(str(animal) + 'completed entropy calculations')
    