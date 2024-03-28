import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 
import EntropyHub as EH

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/DispEn_DF/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1']  


def calculate_dispen(data_input, patient_id, channel_list, clean_indices, results_path):
    
    all_indices = []
    
    for clean_idx in clean_indices:
        all_channels = []
        for chan_idx, channel in enumerate(channel_list):
            Dispx, RDE = EH.DispEn(Sig = np.array(data_input[clean_idx][chan_idx, :]), m=3, tau=2, c=4, Typex='ncdf')
            dispen_dict = {str(channel) + '_dispen': Dispx}
            dispen_df = pd.DataFrame(data = dispen_dict)
            all_channels.append(dispen_df)
        dispen_idx = pd.concat(all_channels, axis = 1)
        all_indices.append(dispen_idx)
    dispen_all_epochs = pd.concat(all_indices, axis = 0)
    
    dispen_all_epochs.to_csv(results_path + str(patient_id) + '_dispen.csv')
    

for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    epoch_array = np.array(epochs)
    
    #load mne features
    patient_mne = pd.read_csv(mne_directory + patient + '_all_conn_measures.csv')
    mne_indices = patient_mne['Idx'].to_list()
    
    # Calculate DispEn for each channel
    channels = ['E1', 'E2', 'F3', 'C3', 'O1', 'M2']
 
    calculate_dispen(data_input = epochs, patient_id = patient,
                         channel_list = channels,
                         clean_indices = mne_indices,
                         results_path = results_path)
    
    print('All channels calculated')
