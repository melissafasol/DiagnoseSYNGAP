import os 
import numpy as np 
import pandas as pd 
import maptlotlib.pyplot as plt

import scipy
from sklearn import mixture
import scipy.stats as stats
import openpyxl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/melissa/PROJECT_DIRECTORIES/EEGFeatureExtraction/Scripts/Preprocessing')
from load_files import LoadFiles
from filter import NoiseFilter, HarmonicsFilter, remove_seizure_epochs
from exploratory import FindNoiseThreshold
from constants import SYNGAP_baseline_start, SYNGAP_baseline_end, channel_variables



def extract_visual_scores_packet_loss(path_to_data, file_name, column_name, target_value):
    '''path_to_data = '/home/melissa/PREPROCESSING
    file_name = 70_noise_validation.xlsx
    column_to_extract = 'B'
    target_value = 1 (if you want to extract the seizures)
    '''
    
    visual_workbook = openpyxl.load_workbook(str(path_to_data) + str(file_name))
    visual_sheet = visual_workbook.active
    column_data = [cell.value for cell in visual_sheet[column_name]]
    
    vis_score_noise = np.array(column_data[1:17281])
    noise_unique, noise_counts = np.unique(vis_score_noise, return_counts=True)
    print('unique values')
    print(noise_unique)
    print('unique value counts')
    print(noise_counts)
    
    def find_indices(arr, target):
        indices = []
        for i, value in enumerate(arr):
            if value == target:
                indices.append(i)
        return indices


    target_value = 1
    noise_indices = find_indices(vis_score_noise, target_value)

    return noise_indices


def remove_typical_packet_loss(packet_loss_dir, file_name):
    
    file = pd.read_pickle(str(packet_loss_dir) + str(file_name))
    packet_loss = file.loc[file['brainstate'] == 6]
    packet_loss_indices = np.array(packet_loss.index)

    return packet_loss_indices


def generate_3d_data(emg_array, eeg_array, noise_indices):
    
    '''calculate features to feed into GMM'''
    
    indices = np.arange(1, 17281, 1)
    
    array_3D_ls = []
    
    for idx, eeg_epoch, emg_epoch in zip(indices, eeg_array, emg_array):
        if idx in noise_indices:
            pass
        else:
            epoch_data_ls = []
            freq, power_emg = scipy.signal.welch(emg_epoch, window='hann', fs=250.4, nperseg=1252)
            freq, power_eeg = scipy.signal.welch(eeg_epoch, window='hann', fs=250.4, nperseg=1252)
        
            num_coefficients = 21
            freq_theta = power_eeg[29:42]
            smoothed_theta_power = np.max(np.log(np.convolve(freq_theta, np.ones(num_coefficients)/num_coefficients, mode='same')))
            
            gamma_eeg = np.mean(np.log(power_eeg[150:240]))
            freq_eeg = np.mean(np.log(power_eeg[5:101]))
            freq_emg = np.mean(np.log(power_emg[300:451]))
            epoch_data_ls.append(smoothed_theta_power)
            epoch_data_ls.append(freq_eeg)
            epoch_data_ls.append(freq_emg)
            epoch_data_ls.append(gamma_eeg)
            array_3D_ls.append(epoch_data_ls)
    
    all_epochs = np.array(array_3D_ls)
    

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler to your data and transform it
    standardized_epochs = scaler.fit_transform(all_epochs)
    
    return standardized_epochs


