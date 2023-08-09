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
results_path = '/home/melissa/RESULTS/XGBoost/Human_SYNGAP1/DispEn'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  = ['P2 N1', 'P2 N2']

        #['P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1','P6 N1', 'P6 N2', 'P7 N1', 'P7 N2', 'P8 N1']
                #['P1 N1', 'P10 N1', 'P11 N1', 'P15 N1', 'P16 N1', 'P17 N1', 'P18 N1', 'P20 N1', 'P21 N1', 'P21 N2', 'P22 N1',
              #'P23 N1', 'P24 N1', 'P28 N1', 'P28 N2', 'P29 N2', 'P30 N1']


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    
    print('data loaded')
    
    #0 index after dispen function because we only want the dispen values and not ppi
    dispen_chan_E1 = [EH.DispEn(epochs[idx][0], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    dispen_chan_E2 = [EH.DispEn(epochs[idx][1], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    dispen_chan_F3 = [EH.DispEn(epochs[idx][2], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    dispen_chan_C3 = [EH.DispEn(epochs[idx][3], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    dispen_chan_O1 = [EH.DispEn(epochs[idx][4], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    dispen_chan_M2 = [EH.DispEn(epochs[idx][5], m = 3, tau = 2, c = 4, Typex = 'ncdf')[0] for idx in clean_indices]
    
    print('all channels calculated')

    os.chdir(results_path)
    np.save(patient + '_E1_dispen.npy', dispen_chan_E1)
    np.save(patient + '_E2_dispen.npy', dispen_chan_E2)
    np.save(patient + '_F3_dispen.npy', dispen_chan_F3)
    np.save(patient + '_C3_dispen.npy', dispen_chan_C3)
    np.save(patient + '_01_dispen.npy', dispen_chan_O1)
    np.save(patient + '_M2_dispen.npy', dispen_chan_M2)
    print(patient + ' saved')