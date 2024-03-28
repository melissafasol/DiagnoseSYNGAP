import os 
import sys

import numpy as np 
import pandas as pd 
import scipy 
import matplotlib
import mne 

from mne_features.univariate import compute_higuchi_fd, compute_hurst_exp

from preprocess_human import load_filtered_data, split_into_epochs, select_clean_indices

human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
results_path = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/HFD_All_Epochs/'
noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'

patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
                  'P28 N2', 'P29 N2', 'P30 N1']  


for patient in patient_list:
    print(patient)
    file_name = patient + '_(1).edf'
    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
    clean_indices = select_clean_indices(noise_directory = noise_directory, patient_id = patient, total_num_epochs = number_epochs)
    
    print('data loaded')
    all_indices = np.arange(0, number_epochs, 1)    
    
    hfd_chan_E1 = [compute_higuchi_fd(np.expand_dims(epochs[idx][0], axis = 0))[0] for idx in clean_indices]
    hfd_chan_E2 = [compute_higuchi_fd(np.expand_dims(epochs[idx][1], axis = 0))[0] for idx in clean_indices]
    hfd_chan_F3 = [compute_higuchi_fd(np.expand_dims(epochs[idx][2], axis = 0))[0] for idx in clean_indices]
    hfd_chan_C3 = [compute_higuchi_fd(np.expand_dims(epochs[idx][3], axis = 0))[0] for idx in clean_indices]
    hfd_chan_O1 = [compute_higuchi_fd(np.expand_dims(epochs[idx][4], axis = 0))[0] for idx in clean_indices]
    hfd_chan_M2 = [compute_higuchi_fd(np.expand_dims(epochs[idx][5], axis = 0))[0] for idx in clean_indices]
    
    print('all channels calculated')

    os.chdir(results_path)
    np.save(patient + '_E1_hfd.npy', hfd_chan_E1)
    np.save(patient + '_E2_hfd.npy', hfd_chan_E2)
    np.save(patient + '_F3_hfd.npy', hfd_chan_F3)
    np.save(patient + '_C3_hfd.npy', hfd_chan_C3)
    np.save(patient + '_01_hfd.npy', hfd_chan_O1)
    np.save(patient + '_M2_hfd.npy', hfd_chan_M2)
    print(patient + ' saved')
    
    
#MNE data 
#patient_list  =  ['P1 N1', 'P2 N1', 'P2 N2', 'P3 N1', 'P3 N2', 'P4 N1', 'P4 N2', 'P5 N1',
#                  'P6 N1', 'P6 N2', 'P7 N1', 'P7 N2','P8 N1','P10 N1', 'P11 N1', 'P15 N1',
#                  'P16 N1', 'P17 N1', 'P18 N1','P20 N1', 'P21 N1', 'P21 N2', 'P21 N3',
#                  'P22 N1','P23 N1', 'P23 N2', 'P23 N3', 'P24 N1','P27 N1','P28 N1',
#                  'P28 N2', 'P29 N2', 'P30 N1']  
#
#mne_directory = '/home/melissa/RESULTS/FINAL_MODEL/Human/Connectivity_MNE/xgb_dataframes/'
#human_data_folder = '/home/melissa/PREPROCESSING/SYNGAP1/SYNGAP1_Human_Data'
#results_path = '/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/HFD_All_Epochs/'
#noise_directory = '/home/melissa/PREPROCESSING/SYNGAP1/human_npy/harmonic_idx/'
#
#for patient in patient_list:
#    print(patient)
#    file_name = patient + '_(1).edf'
#    filtered_data = load_filtered_data(file_path = human_data_folder, file_name = file_name)
#    number_epochs, epochs = split_into_epochs(filtered_data, sampling_rate = 256, num_seconds = 30)
#    epoch_array = np.array(epochs)
#    
#    #load mne features
#    patient_mne = pd.read_csv(mne_directory + patient + '_all_conn_measures.csv')
#    mne_indices = patient_mne['Idx'].to_list()
#    
#    hfd_chan_E1 = [compute_higuchi_fd(np.expand_dims(epochs[idx][0], axis = 0))[0] for idx in mne_indices]
#    hfd_chan_E2 = [compute_higuchi_fd(np.expand_dims(epochs[idx][1], axis = 0))[0] for idx in mne_indices]
#    hfd_chan_F3 = [compute_higuchi_fd(np.expand_dims(epochs[idx][2], axis = 0))[0] for idx in mne_indices]
#    hfd_chan_C3 = [compute_higuchi_fd(np.expand_dims(epochs[idx][3], axis = 0))[0] for idx in mne_indices]
#    hfd_chan_O1 = [compute_higuchi_fd(np.expand_dims(epochs[idx][4], axis = 0))[0] for idx in mne_indices]
#    hfd_chan_M2 = [compute_higuchi_fd(np.expand_dims(epochs[idx][5], axis = 0))[0] for idx in mne_indices]
#    
#    hfd_dict = {'hfd_E1': hfd_chan_E1, 'hfd_E2': hfd_chan_E2, 'hfd_chan_F3': hfd_chan_F3,
#                'hfd_C3': hfd_chan_C3, 'hfd_O1': hfd_chan_O1, 'hfd_chan_M2': hfd_chan_M2} 
#    
#    hfd_df = pd.DataFrame(data = hfd_dict)
#    
#    os.chdir('/home/melissa/RESULTS/FINAL_MODEL/Human/Complexity/hfd_df/')
#    hfd_df.to_csv(patient + '_hfd.csv')